"""
Shared LLM client used by soap_generator, medication_extractor,
decision_support, consultation_summary, and prescription_insights.

Fallback chain: Gemini -> Groq -> Mistral. Each is a genuinely free,
OpenAI-compatible (Groq) or near-compatible (Mistral) text API, used purely
as redundancy against rate limits/overload on any single provider -
callers never know or care which one actually answered.

Note: this fallback is safe for every generator in this codebase EXCEPT
anything that sends video/images directly to a model - none of the
text-generation modules here do that (video analysis, if you build it,
would need its own Gemini-only path, since Groq/Mistral's free-tier chat
models here are text-only).
"""

import json
import re
import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env file in the project root, if present

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Create a .env file (see .env.example) "
        "or set it as an environment variable before starting the server."
    )
# Groq/Mistral are only checked when actually needed (in their call functions
# below), since a missing fallback key shouldn't prevent startup if Gemini
# alone is working fine.


def call_llm(prompt: str, temperature: float = 0.2) -> dict:
    """
    Tries Gemini first (with retry-on-overload), falls back to Groq if
    Gemini is unavailable, then falls back to Mistral if Groq also fails.
    Returns the parsed JSON response from whichever provider succeeds first.
    """
    try:
        return _call_gemini(prompt, temperature)
    except _ProviderUnavailable as gemini_error:
        print(f"Gemini unavailable ({gemini_error}) - falling back to Groq...")
        try:
            return _call_openai_compatible(GROQ_URL, GROQ_API_KEY, GROQ_MODEL, prompt, temperature)
        except Exception as groq_error:
            print(f"Groq also failed ({groq_error}) - falling back to Mistral...")
            return _call_openai_compatible(MISTRAL_URL, MISTRAL_API_KEY, MISTRAL_MODEL, prompt, temperature)


class _ProviderUnavailable(Exception):
    """Raised when Gemini fails due to transient overload (503) or hitting
    its rate/quota limit (429) - triggers the fallback chain. Any other
    Gemini error (bad key, invalid request, etc.) still raises normally,
    since falling back wouldn't fix those either."""
    pass


def _call_gemini(prompt: str, temperature: float) -> dict:
    max_retries = 2  # fewer retries here since the fallback chain covers the rest
    response = None
    for attempt in range(1, max_retries + 1):
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": temperature,
                },
            },
        )
        if response.status_code == 200:
            break

        is_overloaded = response.status_code == 503 or "UNAVAILABLE" in response.text
        is_rate_limited = response.status_code == 429 or "RESOURCE_EXHAUSTED" in response.text

        if is_overloaded or is_rate_limited:
            if attempt < max_retries and is_overloaded:
                time.sleep(attempt * 4)
                continue
            raise _ProviderUnavailable(f"status {response.status_code}")

        raise RuntimeError(f"Gemini error {response.status_code}: {response.text}")

    data = response.json()
    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_json_response(raw_text)


def _call_openai_compatible(url: str, api_key: str, model: str, prompt: str, temperature: float) -> dict:
    """Shared caller for both Groq and NVIDIA NIM, since both expose an
    OpenAI-compatible chat completions endpoint with identical request/
    response shape - only the URL, key, and model name differ."""
    if not api_key:
        raise RuntimeError(f"No API key set for {url} - add it to your .env file.")

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
    )

    if response.status_code != 200:
        raise RuntimeError(f"{url} error {response.status_code}: {response.text}")

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"]
    return _parse_json_response(raw_text)


def _parse_json_response(raw_text: str) -> dict:
    """
    Parses the LLM's JSON output, with fallbacks for common malformations:
    markdown fences, trailing data after a complete object, unescaped quotes,
    and literal (unescaped) newlines inside string values.
    """
    raw_text = re.sub(r"```json|```", "", raw_text).strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(raw_text)
            return parsed

        # Try the json_repair library first - it robustly handles unescaped
        # newlines, quotes, trailing commas, and other common LLM JSON
        # malformations far better than a hand-rolled regex can.
        try:
            from json_repair import repair_json
            repaired = repair_json(raw_text)
            return json.loads(repaired)
        except Exception:
            pass

        # Last-resort fallback: attempt to repair unescaped quotes manually
        # (covers the case where json_repair isn't installed).
        repaired = re.sub(r'(?<!\\)"(?!\s*[,:}\]])', r'\\"', raw_text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse LLM response as JSON: {e}. Raw: {raw_text[:500]}")