"""
LLM abstraction layer — supports Anthropic Claude and Google Gemini.
Uses: anthropic SDK + google-genai SDK (new, replaces deprecated google-generativeai).
"""
import re
import time
import anthropic
from google import genai
from google.genai import types as genai_types
from config import ANTHROPIC_API_KEY, GEMINI_API_KEY

# Models
CLAUDE_FAST_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_DEEP_MODEL = "claude-sonnet-4-6"
GEMINI_FAST_MODEL = "gemini-2.5-flash-lite"
GEMINI_DEEP_MODEL = "gemini-2.5-flash-lite"

# Gemini rate limiter — free tier allows 15 RPM; we cap at 12 to stay safe
_GEMINI_RPM_LIMIT = 12
_gemini_call_times: list[float] = []


def _gemini_rate_limit():
    """Block until making another Gemini call won't exceed the per-minute limit."""
    now = time.time()
    # Drop timestamps older than 60 s
    _gemini_call_times[:] = [t for t in _gemini_call_times if now - t < 60]
    if len(_gemini_call_times) >= _GEMINI_RPM_LIMIT:
        wait = 60 - (now - _gemini_call_times[0]) + 0.5
        if wait > 0:
            print(f"  [Gemini rate limiter] waiting {wait:.1f}s …")
            time.sleep(wait)
    _gemini_call_times.append(time.time())


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes add around JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def call_llm(
    prompt: str,
    system: str = "",
    provider: str = "claude",
    mode: str = "fast",
    max_tokens: int = 2000,
) -> str:
    """
    Unified LLM call. Returns raw text response.
    Raises RuntimeError on auth/quota failure.

    provider : "claude" | "gemini"
    mode     : "fast" (bulk/cheap) | "deep" (quality thesis)
    """
    if provider == "gemini":
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
            raise RuntimeError("GEMINI_API_KEY is not set in your .env file.")
        return _call_gemini(prompt, system, mode, max_tokens)

    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
        raise RuntimeError("ANTHROPIC_API_KEY is not set in your .env file.")
    return _call_claude(prompt, system, mode, max_tokens)


def _call_claude(prompt: str, system: str, mode: str, max_tokens: int) -> str:
    model = CLAUDE_FAST_MODEL if mode == "fast" else CLAUDE_DEEP_MODEL
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except anthropic.AuthenticationError:
        raise RuntimeError("Invalid ANTHROPIC_API_KEY. Check your .env file.")
    except anthropic.RateLimitError:
        raise RuntimeError("Anthropic rate limit hit. Wait a minute and retry.")
    except Exception as e:
        raise RuntimeError(f"Claude API error: {e}")


def _call_gemini(prompt: str, system: str, mode: str, max_tokens: int) -> str:
    model_name = GEMINI_FAST_MODEL if mode == "fast" else GEMINI_DEEP_MODEL
    _gemini_rate_limit()
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        config = genai_types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(str(e))


def provider_label(provider: str, mode: str) -> str:
    labels = {
        ("claude", "fast"): "Claude Haiku 4.5 (~$0.001/call)",
        ("claude", "deep"): "Claude Sonnet 4.6 (~$0.01/call)",
        ("gemini", "fast"): "Gemini 2.5 Flash Lite (free tier, 1500 req/day)",
        ("gemini", "deep"): "Gemini 2.5 Flash Lite (free tier, 1500 req/day)",
    }
    return labels.get((provider, mode), f"{provider}/{mode}")
