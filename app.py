import os
import asyncio
from flask import Flask, render_template, request, jsonify
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from dotenv import load_dotenv
from collections import Counter
import re
import string

load_dotenv()

app = Flask(__name__)

anthropic_client = AsyncAnthropic()
openai_client = AsyncOpenAI()


@app.route("/")
def index():
    return render_template("index.html")


def normalize_response(text, options):
    """Normalize response text based on selected options."""
    result = text.strip()

    if options.get("strip_punctuation"):
        result = result.strip(string.punctuation)

    if options.get("lowercase"):
        result = result.lower()

    if options.get("first_word_only"):
        result = result.split()[0] if result.split() else result
        if options.get("strip_punctuation"):
            result = result.strip(string.punctuation)

    if options.get("normalize_whitespace"):
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'[–—−]', '-', result)

    return result


async def call_openai(model, temperature, system_message, prompt):
    """Make a single OpenAI API call."""
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {str(e)}]"


async def call_anthropic(model, temperature, system_message, prompt):
    """Make a single Anthropic API call."""
    try:
        message = await anthropic_client.messages.create(
            model=model,
            max_tokens=8192,
            temperature=temperature,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text.strip()
    except Exception as e:
        return f"[ERROR: {str(e)}]"


async def run_parallel_requests(model, temperature, prompt, iterations, is_openai):
    """Run all API requests in parallel."""
    system_message = "You are a helpful assistant. Follow the user's instructions precisely. Be concise."

    if is_openai:
        tasks = [
            call_openai(model, temperature, system_message, prompt)
            for _ in range(iterations)
        ]
    else:
        tasks = [
            call_anthropic(model, temperature, system_message, prompt)
            for _ in range(iterations)
        ]

    return await asyncio.gather(*tasks)


@app.route("/test", methods=["POST"])
def test_prompt():
    data = request.json
    prompt = data.get("prompt", "")
    iterations = int(data.get("iterations", 10))
    model = data.get("model", "gpt-4o-mini")
    temperature = float(data.get("temperature", 1.0))

    normalize_options = {
        "strip_punctuation": True,
        "lowercase": True,
        "normalize_whitespace": True,
        "first_word_only": data.get("first_word_only", False),
    }

    iterations = max(1, min(500, iterations))
    temperature = max(0, min(1, temperature))

    is_openai = model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4")

    raw_results = asyncio.run(run_parallel_requests(model, temperature, prompt, iterations, is_openai))

    results = [normalize_response(text, normalize_options) for text in raw_results]

    counts = Counter(results)
    chart_data = {
        "labels": list(counts.keys()),
        "values": list(counts.values())
    }

    unique_count = len(counts)
    most_common = counts.most_common(1)[0] if counts else ("N/A", 0)

    return jsonify({
        "results": results,
        "chart_data": chart_data,
        "stats": {
            "total": len(results),
            "unique": unique_count,
            "most_common": most_common[0],
            "most_common_count": most_common[1],
            "consistency": round((most_common[1] / len(results)) * 100, 1) if results else 0
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
