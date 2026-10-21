
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

async def main():
    client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    response = await client.chat.completions.create(
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        ),
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: GROQ CONNECTION SUCCESS",
            }
        ],
        temperature=0,
    )

    print(response.choices[0].message.content)

asyncio.run(main())
