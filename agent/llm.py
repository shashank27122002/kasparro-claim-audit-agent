
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

from agent.schemas import Claim, Prediction


load_dotenv(override=True)

DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_API_ATTEMPTS = 3
MAX_RETRY_DELAY_SECONDS = 90
MAX_FACT_RESULT_CHARS = 6000


SYSTEM_PROMPT = """
Audit factual claims in the answer using catalogue evidence.
Return ONLY compact valid JSON matching:
{"claims":[{"quote":"exact source substring","label":"supported|contradicted|unknown","evidence_fact_ids":["actual_fact_id"],"explanation":"brief reason"}]}

Rules:
- Extract independently verifiable factual claims; quote each verbatim and exactly.
- Use get_brand_facts when needed; only cite IDs actually returned by the tool.
- supported = evidence affirms the precise claim; contradicted = evidence directly conflicts; unknown = insufficient evidence.
- Missing evidence is not proof of falsity. Never invent facts or IDs.
- Consider brand/model/variant, market, date, units, terms, exclusions, and conditions.
- Keep explanations very short (one short phrase, ideally under 10 words).
- CRITICAL: quote must be copied character-for-character from ANSWER below.
- Never put reasoning, inference, or a paraphrase in the quote field.
- Before returning JSON, verify answer_text.find(quote) would succeed for every quote.
- If a proposed claim cannot be quoted exactly, omit that claim rather than inventing a quote.
- Do not add markdown or text outside JSON. Include every required field.
"""


SYSTEM_PROMPT_V2 = """
Strict evidence-grounded claim auditor. Return ONLY compact valid JSON:
{"claims":[{"quote":"exact source substring","label":"supported|contradicted|unknown","evidence_fact_ids":["actual_fact_id"],"explanation":"brief reason"}]}

Extract independently verifiable factual claims. Quotes must be exact literal substrings, preserving punctuation and numbers; never merge non-contiguous text.
Use catalogue facts for the exact brand/model. Cite only IDs actually returned. supported = evidence affirms the precise claim; contradicted = evidence directly conflicts; unknown = insufficient evidence. Missing evidence is not proof of falsity.
Check variant, market, date, units, warranty scope, purchase terms, exclusions, and testing conditions. Never invent facts, IDs, or citations. Keep explanations to one short phrase (ideally under 10 words). CRITICAL: every quote must be copied character-for-character from ANSWER; never put reasoning or a paraphrase in quote. Verify each quote is a literal substring before returning JSON. If a claim cannot be quoted exactly, omit it. No markdown or text outside JSON. Include every required field.
"""


class Agent:
    def __init__(
        self,
        facts: Any,
        model: str | None = None,
        max_attempts: int = DEFAULT_API_ATTEMPTS,
        max_tool_rounds: int = 1,
        max_tokens: int = 2048,
        concurrency: int = 1,
    ) -> None:
        self.facts = facts
        self.model = (
            model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
        )

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Set it in your local .env file."
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("GROQ_BASE_URL", GROQ_BASE_URL),
            max_retries=0,
            timeout=45.0,
        )

        self.max_attempts = max(1, max_attempts)
        self.max_tool_rounds = 1
        self.max_tokens = max(1, max_tokens)
        self._semaphore = asyncio.Semaphore(max(1, concurrency))

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_brand_facts",
                    "description": (
                        "Retrieve authoritative catalogue facts for the "
                        "specified brand and a focused claim or model query."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "brand_id": {
                                "type": "string",
                                "description": (
                                    "Brand ID from the current answer row."
                                ),
                            },
                            "query": {
                                "type": "string",
                                "description": (
                                    "Claim, model, or catalogue detail to find."
                                ),
                            },
                        },
                        "required": ["brand_id", "query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    # ---------------------------------------------------------
    # Row helpers
    # ---------------------------------------------------------

    @staticmethod
    def _row_value(
        row: Any,
        key: str,
        default: Any = "",
    ) -> Any:
        """Read values from dictionaries, pandas Series, or objects."""
        if isinstance(row, dict):
            return row.get(key, default)

        if hasattr(row, "get"):
            try:
                value = row.get(key, default)
                return default if value is None else value
            except (TypeError, AttributeError):
                pass

        return getattr(row, key, default)

    # ---------------------------------------------------------
    # Rate-limit handling
    # ---------------------------------------------------------

    @staticmethod
    def _is_daily_quota_error(exc: RateLimitError) -> bool:
        message = str(exc).lower()

        return any(
            marker in message
            for marker in (
                "tokens per day",
                "requests per day",
                "daily limit",
                "per day (tpd)",
            )
        )

    @staticmethod
    def _rate_limit_delay(
        exc: RateLimitError,
    ) -> float | None:
        # Prefer the exact wait time stated in the provider error.
        match = re.search(
            r"try again in\s+([\d.]+)\s*s",
            str(exc),
            flags=re.IGNORECASE,
        )

        if match:
            return float(match.group(1)) + 1.0

        response = getattr(exc, "response", None)
        headers = (
            getattr(response, "headers", None)
            if response is not None
            else None
        )

        if headers:
            value = headers.get("retry-after")

            if value:
                try:
                    return max(1.0, float(value)) + 1.0
                except (TypeError, ValueError):
                    pass

        return None

    async def _api_request(self, **kwargs: Any) -> Any:
        """Retry an individual API request without repeating earlier calls."""
        last_error: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                return await self.client.chat.completions.create(
                    **kwargs
                )

            except RateLimitError as exc:
                last_error = exc

                # Waiting does not fix an exhausted daily quota.
                if self._is_daily_quota_error(exc):
                    raise

                if attempt + 1 >= self.max_attempts:
                    raise

                delay = self._rate_limit_delay(exc)

                if delay is None:
                    delay = min(
                        5 * (2 ** attempt),
                        MAX_RETRY_DELAY_SECONDS,
                    )

                delay = min(
                    max(1.0, delay),
                    MAX_RETRY_DELAY_SECONDS,
                )

                print(
                    f"Groq rate limit for {self.model}; "
                    f"attempt {attempt + 1}/{self.max_attempts}. "
                    f"Waiting {delay:.1f}s.",
                    flush=True,
                )

                await asyncio.sleep(delay)

            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc

                if attempt + 1 >= self.max_attempts:
                    raise

                delay = min(2 ** attempt, 30)

                print(
                    f"Temporary connection error; retrying in {delay}s.",
                    flush=True,
                )

                await asyncio.sleep(delay)

            except APIError:
                # Do not retry unrelated API errors indefinitely.
                raise

        if last_error is not None:
            raise last_error

        raise RuntimeError("API request failed without an error.")

    # ---------------------------------------------------------
    # Catalogue facts
    # ---------------------------------------------------------

    async def _call_facts_tool(
        self,
        arguments: dict[str, Any],
        expected_brand_id: str,
    ) -> Any:
        query = arguments.get("query", "")
        brand_id = expected_brand_id.strip()

        # Always use the dataset's brand ID.
        if not brand_id:
            return {"error": "The answer has no brand_id."}

        if not isinstance(query, str) or not query.strip():
            return {"error": "A non-empty query is required."}

        method = getattr(self.facts, "get_brand_facts", None)

        if not callable(method):
            raise RuntimeError(
                "The facts provider does not implement get_brand_facts()."
            )

        result = method(
            brand_id=brand_id,
            query=query.strip(),
        )

        if hasattr(result, "__await__"):
            result = await result

        if result is None:
            result = {"facts": []}
        elif hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        elif not isinstance(
            result,
            (dict, list, str, int, float, bool),
        ):
            result = {"result": str(result)}

        serialized = json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

        if len(serialized) > MAX_FACT_RESULT_CHARS:
            return {
                "warning": (
                    "Catalogue response exceeded the size limit. "
                    "Use only complete facts and IDs actually present "
                    "in the returned data. If evidence is insufficient, "
                    "classify the claim as unknown."
                ),
                "truncated_catalogue_json": serialized[
                    :MAX_FACT_RESULT_CHARS
                ],
            }

        return result

    # ---------------------------------------------------------
    # JSON parsing
    # ---------------------------------------------------------

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("The model returned empty JSON content.")

        text = content.strip()

        text = re.sub(
            r"^\s*```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```\s*$", "", text)

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")

            if start < 0 or end <= start:
                raise ValueError(
                    "The model response did not contain a JSON object."
                )

            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON returned by the model: {exc}"
                ) from exc

        if not isinstance(payload, dict):
            raise ValueError("The model output must be a JSON object.")

        if not isinstance(payload.get("claims"), list):
            raise ValueError(
                "The model JSON must contain a claims array."
            )

        return payload

    # ---------------------------------------------------------
    # Prediction validation
    # ---------------------------------------------------------

    @staticmethod
    def _find_quote_span(answer_text: str, quote: str) -> tuple[int, int] | None:
        """Find exact text, or tolerate dash Unicode variants only.

        The caller stores the exact source slice, preserving source offsets.
        """
        exact_start = answer_text.find(quote)
        if exact_start >= 0:
            return exact_start, exact_start + len(quote)

        dash_variants = {
            "\u2010", "\u2011", "\u2012", "\u2013",
            "\u2014", "\u2212", "\ufe63", "\uff0d",
        }

        def normalize_dash(char: str) -> str:
            return "-" if char in dash_variants else char

        normalized_answer = "".join(normalize_dash(char) for char in answer_text)
        normalized_quote = "".join(normalize_dash(char) for char in quote)
        normalized_start = normalized_answer.find(normalized_quote)

        if normalized_start < 0:
            return None

        return normalized_start, normalized_start + len(normalized_quote)

    def _build_prediction(
        self,
        payload: dict[str, Any],
        answer_text: str,
        answer_id: str,
        version: str,
    ) -> Prediction:
        claims: list[Claim] = []

        for index, item in enumerate(payload["claims"]):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Claim at index {index} must be a JSON object."
                )

            quote = item.get("quote")
            label = item.get("label")
            evidence_ids = item.get("evidence_fact_ids", [])
            explanation = item.get("explanation", "")

            if not isinstance(quote, str) or not quote:
                raise ValueError(
                    f"Claim at index {index} needs a non-empty quote."
                )

            # Prefer exact matching. If only dash Unicode differs, replace
            # the model quote with the exact source slice before storing it.
            quote_span = self._find_quote_span(answer_text, quote)

            if quote_span is None:
                raise ValueError(
                    "Claim quote is not an exact substring of the answer "
                    "(even after dash-only matching): "
                    f"{quote!r}"
                )

            start, end = quote_span
            quote = answer_text[start:end]

            if label not in {
                "supported",
                "contradicted",
                "unknown",
            }:
                raise ValueError(
                    f"Unsupported claim label: {label!r}"
                )

            if not isinstance(evidence_ids, list) or not all(
                isinstance(value, str) for value in evidence_ids
            ):
                raise ValueError(
                    "evidence_fact_ids must be a list of strings."
                )

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

    # ---------------------------------------------------------
    # Model request
    # ---------------------------------------------------------

    async def _request_prediction(
        self,
        answer_text: str,
        answer_id: str,
        brand_id: str,
        version: str,
        retry_feedback: str | None = None,
    ) -> Prediction:
        if version not in {"v1", "v2"}:
            raise ValueError(
                f"Unsupported agent version: {version}"
            )

        system_prompt = (
            SYSTEM_PROMPT_V2
            if version == "v2"
            else SYSTEM_PROMPT
        )

        user_content = (
            f"Answer ID: {answer_id}\n"
            f"Brand ID: {brand_id}\n\n"
            "Audit the answer below. If catalogue evidence is needed, "
            "use get_brand_facts with the exact brand ID shown above. "
            "After receiving catalogue facts, return final JSON. "
            "Do not repeat the same lookup.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Copy quotes exactly from the original answer.\n"
            "- Never paraphrase or invent a quote.\n"
            "- Never invent catalogue facts or evidence IDs.\n"
            "- Use unknown when evidence is insufficient.\n"
            "- The quote field must contain only an exact, contiguous substring from ANSWER.\n"
            "- Never put a paraphrase, inference, or explanatory sentence in quote.\n"
            "- Verify each quote is present character-for-character before returning JSON.\n"
            "- If no exact quote can be selected for a claim, omit that claim.\n"
            "- Keep each explanation under 10 words where possible.\n\n"
        )

        if retry_feedback:
            user_content += (
                "CORRECTION REQUIRED:\n"
                f"{retry_feedback}\n"
                "Return complete valid JSON. The quote-validation failure must be fixed. "
                "Each quote must be copied verbatim as a contiguous substring of the exact ANSWER text. "
                "Do not put a paraphrase, inference, or explanation in quote. "
                "Omit a claim if you cannot quote it exactly. Keep explanations very short.\n\n"
            )

        user_content += f"ANSWER:\n{answer_text}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Initial request: tool calls are permitted.
        response = await self._api_request(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=0,
            max_tokens=self.max_tokens,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length" and not message.tool_calls:
            raise ValueError(
                "Initial model response was truncated by the output "
                "token limit. Reduce the prompt or increase max_tokens "
                "if the available token budget permits."
            )

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
                        "error": f"Unknown tool: {tool_call.function.name}"
                    }
                else:
                    try:
                        arguments = json.loads(
                            tool_call.function.arguments or "{}"
                        )

                        if not isinstance(arguments, dict):
                            raise ValueError(
                                "Tool arguments must be a JSON object."
                            )

                        tool_result = await self._call_facts_tool(
                            arguments=arguments,
                            expected_brand_id=brand_id,
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

            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The catalogue lookup is complete. Do not call "
                        "any more tools. Return the final JSON now. "
                        "Use only evidence IDs actually present in the "
                        "tool results. If evidence is insufficient, use "
                        "unknown. Return a complete concise response."
                    ),
                }
            )

            # Final request: tools are disabled. JSON mode is used here
            # rather than the tool-enabled request for compatibility.
            response = await self._api_request(
                model=self.model,
                messages=messages,
                tool_choice="none",
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=self.max_tokens,
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

        else:
            # The model returned its JSON directly without a tool call.
            # Make a JSON-mode final request only if the first response
            # was not already a complete valid prediction.
            if message.content:
                try:
                    payload = self._parse_json(message.content)
                    return self._build_prediction(
                        payload=payload,
                        answer_text=answer_text,
                        answer_id=answer_id,
                        version=version,
                    )
                except ValueError:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.content,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was invalid. "
                                "Return a complete JSON object with a "
                                "'claims' array, following the required "
                                "schema. Do not call tools. Every quote must be copied "
                                "character-for-character from ANSWER and be a contiguous substring. "
                                "Never use a paraphrase in quote; omit claims without exact quotes. "
                                "Keep explanations short."
                            ),
                        }
                    )

            response = await self._api_request(
                model=self.model,
                messages=messages,
                tool_choice="none",
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=self.max_tokens,
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            # One bounded recovery attempt: retain the same evidence/tool
            # context, request compact explanations, and raise the output
            # allowance only for this response. This avoids repeating the
            # initial catalogue lookup when only the final JSON was truncated.
            retry_messages = list(messages)
            retry_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Your previous final JSON was cut off. Regenerate the "
                        "ENTIRE JSON object from the beginning. Keep every "
                        "required claims and quotes copied from ANSWER. "
                        "Keep each explanation a short phrase. Copy quote text "
                        "verbatim, including punctuation and Unicode characters. "
                        "Do not invent evidence IDs or add markdown."
                    ),
                }
            )
            retry_limit = min(max(self.max_tokens + 1024, 3072), 4096)
            print(
                f"Final JSON was truncated; retrying once with "
                f"max_tokens={retry_limit}.",
                flush=True,
            )
            response = await self._api_request(
                model=self.model,
                messages=retry_messages,
                tool_choice="none",
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=retry_limit,
            )
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise ValueError(
                    "Model response remained truncated after one compact "
                    f"retry at {retry_limit} output tokens. The checkpoint "
                    "is preserved; provider TPM limits may prevent recovery."
                )

        if not message.content:
            raise ValueError("The model returned an empty response.")

        payload = self._parse_json(message.content)

        return self._build_prediction(
            payload=payload,
            answer_text=answer_text,
            answer_id=answer_id,
            version=version,
        )

    # ---------------------------------------------------------
    # Public prediction method
    # ---------------------------------------------------------

    async def predict(
        self,
        row: Any,
        version: str = "v1",
    ) -> Prediction:
        if version not in {"v1", "v2"}:
            raise ValueError(
                f"Unsupported agent version: {version}"
            )

        answer_id = str(
            self._row_value(row, "answer_id", "unknown")
        )

        answer_text = str(
            self._row_value(row, "answer_text", "") or ""
        )

        brand_id = str(
            self._row_value(row, "brand_id", "") or ""
        ).strip()

        if not answer_text.strip():
            raise ValueError(
                f"Answer {answer_id} has empty answer_text."
            )

        if not brand_id:
            raise ValueError(
                f"Answer {answer_id} has an empty brand_id."
            )

        async with self._semaphore:
            retry_feedback: str | None = None

            # Validation retries are deliberately bounded. Rate limits
            # are retried separately by _api_request().
            for attempt in range(2):
                try:
                    return await self._request_prediction(
                        answer_text=answer_text,
                        answer_id=answer_id,
                        brand_id=brand_id,
                        version=version,
                        retry_feedback=retry_feedback,
                    )

                except ValueError as exc:
                    if attempt == 1:
                        raise

                    retry_feedback = str(exc)

                    print(
                        f"Invalid model output for {answer_id}; "
                        "retrying once with validation feedback.",
                        flush=True,
                    )

            raise RuntimeError(
                f"Prediction failed for answer {answer_id}."
            )
