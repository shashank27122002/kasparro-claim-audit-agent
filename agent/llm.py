
import asyncio
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError

from agent.schemas import Claim, Prediction

load_dotenv(override=False)


# Baseline prompt: preserve the original approach for comparison.
SYSTEM_PROMPT = """
You are a factual claim-auditing agent.

Extract atomic, independently verifiable factual claims from the
provided answer. Use the catalogue facts tool to check claims.

Return ONLY valid JSON:
{
  "claims": [
    {
      "quote": "Exact text copied from answer",
      "label": "supported",
      "evidence_fact_ids": ["actual_catalogue_fact_id"],
      "explanation": "Brief explanation"
    }
  ]
}

Rules:
- Labels must be supported, contradicted, or unknown.
- Quotes must be exact text copied from the answer.
- Never invent facts or fact IDs.
- Use only fact IDs returned by the catalogue tool.
- If the facts do not establish a claim, label it unknown.
- Missing information does not prove a claim false.
- Keep claims atomic.
- Do not output character offsets; the application calculates them.
- Do not output Markdown fences or text outside the JSON object.
- If there are no factual claims, return {"claims": []}.
"""


# Improved prompt: stricter evidence grounding and source-span accuracy.
SYSTEM_PROMPT_V2 = """
You are a conservative factual claim-auditing agent.

Extract independently verifiable factual claims from the supplied answer.
Use the catalogue facts tool and base every verdict exclusively on the
facts actually returned by that tool.

Return ONLY valid JSON:
{
  "claims": [
    {
      "quote": "Exact text copied from answer",
      "label": "supported",
      "evidence_fact_ids": ["actual_catalogue_fact_id"],
      "explanation": "Brief evidence-based explanation"
    }
  ]
}

Rules:
1. Split sentences containing multiple independently verifiable claims
   into atomic claims.
2. Every quote must be an exact, contiguous substring of the original
   answer, preserving wording and punctuation.
3. Use supported only when catalogue evidence directly substantiates
   the claim.
4. Use contradicted only when catalogue evidence directly conflicts
   with the claim.
5. Use unknown when the catalogue evidence is insufficient to decide.
6. Never treat missing information as proof of falsity.
7. Cite only relevant fact IDs actually returned by the tool.
8. Use the smallest sufficient set of evidence fact IDs.
9. Explain briefly how the evidence supports, contradicts, or fails to
   establish the claim.
10. Do not rely on general knowledge, assumptions, or invented details.
11. Do not infer product specifications, prices, or availability without
    explicit supporting evidence.
12. Never rewrite or paraphrase a quote.
13. Do not output character offsets; the application calculates them.
14. Output valid JSON only, without Markdown fences.
15. If there are no factual claims, return {"claims": []}.
"""


def _get_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


class Agent:
    def __init__(self, facts: Any):
        self.facts = facts

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Configure it in your local .env file."
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://api.groq.com/openai/v1",
            ),
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "90")),
            max_retries=0,
        )

        self.max_attempts = _get_env_int("LLM_MAX_ATTEMPTS", 3)
        self.max_tool_rounds = _get_env_int("LLM_MAX_TOOL_ROUNDS", 5)
        self.max_tokens = _get_env_int("LLM_MAX_TOKENS", 4096)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_brand_facts",
                    "description": (
                        "Retrieve catalogue facts relevant to a brand or "
                        "factual claim. Use these facts as the evidence "
                        "source for claim verification."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Brand, product, attribute, or factual "
                                    "question to look up."
                                ),
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

        self._semaphore = asyncio.Semaphore(
            _get_env_int("LLM_CONCURRENCY", 1)
        )

    async def _call_facts_tool(self, arguments: dict) -> Any:
        """Call the existing catalogue lookup without blocking the event loop."""
        query = arguments.get("query", "")

        if not isinstance(query, str) or not query.strip():
            return {"error": "A non-empty query is required."}

        try:
            result = await asyncio.to_thread(
                self.facts.get_brand_facts,
                query=query,
            )
        except TypeError:
            # Support existing implementations that accept positional input.
            result = await asyncio.to_thread(
                self.facts.get_brand_facts,
                query,
            )

        if hasattr(result, "model_dump"):
            return result.model_dump()

        if hasattr(result, "to_dict"):
            return result.to_dict()

        if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
            return result

        return str(result)

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Parse JSON, allowing a defensive fallback for fenced responses."""
        content = content.strip()

        if content.startswith("```"):
            content = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                content,
                flags=re.IGNORECASE,
            ).strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")

            if start < 0 or end <= start:
                raise ValueError("The model response did not contain valid JSON.")

            parsed = json.loads(content[start : end + 1])

        if not isinstance(parsed, dict):
            raise ValueError("The model JSON response must be an object.")

        claims = parsed.get("claims")

        if not isinstance(claims, list):
            raise ValueError("The model response must contain a claims list.")

        return parsed

    def _build_prediction(
        self,
        payload: dict,
        answer_text: str,
        answer_id: str,
        version: str,
    ) -> Prediction:
        claims = []

        for item in payload["claims"]:
            if not isinstance(item, dict):
                raise ValueError("Each claim must be a JSON object.")

            quote = item.get("quote")
            label = item.get("label")
            evidence_ids = item.get("evidence_fact_ids", [])
            explanation = item.get("explanation", "")

            if not isinstance(quote, str) or not quote:
                raise ValueError("Every claim must contain a non-empty quote.")

            # Exact substring validation prevents fabricated source spans.
            start = answer_text.find(quote)

            if start < 0:
                raise ValueError(
                    f"Claim quote is not an exact substring of the answer: "
                    f"{quote!r}"
                )

            if label not in {"supported", "contradicted", "unknown"}:
                raise ValueError(f"Unsupported claim label: {label!r}")

            if not isinstance(evidence_ids, list) or not all(
                isinstance(value, str) for value in evidence_ids
            ):
                raise ValueError("evidence_fact_ids must be a list of strings.")

            if not isinstance(explanation, str):
                explanation = str(explanation)

            claims.append(
                Claim(
                    quote=quote,
                    start_char=start,
                    end_char=start + len(quote),
                    label=label,
                    evidence_fact_ids=evidence_ids,
                    explanation=explanation,
                )
            )

        return Prediction(
            answer_id=answer_id,
            version=version,
            claims=claims,
        )

    async def _request_prediction(
        self,
        answer_text: str,
        answer_id: str,
        version: str,
    ) -> Prediction:
        if version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported agent version: {version}")

        # The two versions deliberately use different system prompts.
        system_prompt = (
            SYSTEM_PROMPT_V2 if version == "v2" else SYSTEM_PROMPT
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Answer ID: {answer_id}\n\n"
                    "Audit the answer below. Retrieve catalogue facts "
                    "when necessary, then return the required JSON.\n\n"
                    f"ANSWER:\n{answer_text}"
                ),
            },
        ]

        for _ in range(self.max_tool_rounds):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0,
                max_tokens=self.max_tokens,
            )

            message = response.choices[0].message

            if message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            call.model_dump(exclude_none=True)
                            for call in message.tool_calls
                        ],
                    }
                )

                for tool_call in message.tool_calls:
                    if tool_call.function.name != "get_brand_facts":
                        tool_result = {
                            "error": (
                                f"Unknown tool: {tool_call.function.name}"
                            )
                        }
                    else:
                        try:
                            arguments = json.loads(
                                tool_call.function.arguments or "{}"
                            )
                            tool_result = await self._call_facts_tool(
                                arguments
                            )
                        except Exception as exc:
                            tool_result = {
                                "error": (
                                    f"Catalogue lookup failed: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                            }

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                tool_result,
                                ensure_ascii=False,
                                default=str,
                            ),
                        }
                    )

                continue

            if not message.content:
                raise ValueError("The model returned an empty response.")

            payload = self._parse_json(message.content)

            return self._build_prediction(
                payload=payload,
                answer_text=answer_text,
                answer_id=answer_id,
                version=version,
            )

        raise RuntimeError(
            f"Exceeded {self.max_tool_rounds} tool-call rounds "
            f"for answer {answer_id}."
        )

    async def predict(self, row: Any, version: str = "v1") -> Prediction:
        if version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported agent version: {version}")

        answer_id = str(getattr(row, "answer_id", "unknown"))
        answer_text = str(getattr(row, "answer_text", ""))

        if not answer_text.strip():
            raise ValueError(f"Answer {answer_id} has empty answer_text.")

        async with self._semaphore:
            last_error = None

            for attempt in range(self.max_attempts):
                try:
                    return await self._request_prediction(
                        answer_text=answer_text,
                        answer_id=answer_id,
                        version=version,
                    )
                except RateLimitError:
                    # Do not repeatedly hammer a depleted provider quota.
                    raise
                except (APIError, ValueError, RuntimeError) as exc:
                    last_error = exc

                    if attempt + 1 >= self.max_attempts:
                        break

                    delay = min(2 ** attempt, 8)
                    print(
                        f"Retrying {answer_id} "
                        f"({attempt + 1}/{self.max_attempts}) after "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    await asyncio.sleep(delay)

            raise RuntimeError(
                f"Prediction failed after {self.max_attempts} attempts: "
                f"{last_error}"
            ) from last_error
