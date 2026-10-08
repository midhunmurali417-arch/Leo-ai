"""
=============================================================================
Module: evaluation/providers.py
Description: Minimal provider execution and failover logic mirrored from app.py.
             Allows evaluation scripts to run LLM streaming, detect rate limits,
             and benchmark models WITHOUT importing the Streamlit UI.
=============================================================================
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Generator, Optional, Any
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core import exceptions
import openai
import anthropic

# 1. System Prompt matching app.py
def get_leo_system_prompt() -> str:
    """Generates the temporal-anchored system prompt identical to app.py."""
    now_str = datetime.now().strftime("%A, %B %d, %Y")
    return (
        f"You are Leo 🦁, a highly intelligent, accurate, articulate, and friendly cartoon AI lion assistant.\n"
        f"Temporal Anchor: Today's current date is {now_str}.\n\n"
        f"CORE DIRECTIVES FOR ACCURACY AND TRUTHFULNESS:\n"
        f"1. Factual Rigor: Prioritize precision, verified facts, and sound logical reasoning above all else. Never fabricate, invent, or hallucinate names, dates, scores, syntax, or citations.\n"
        f"2. Temporal Grounding: Whenever answering questions about 'current', 'latest', 'today', or 'this year', reason strictly using the current anchor date ({now_str}).\n"
        f"3. Google Search Grounding: When live Google Search results and real-time news feeds are provided in your context, integrate and synthesize them to answer queries about breaking news, current events, recent tech releases, and real-time facts. Always cite source names (e.g. BBC, Reuters, TechCrunch) and publication dates when available.\n"
        f"4. Knowledge Boundaries: If real-time search context is unavailable for a query occurring after your training cutoff, honestly state what you know and clarify that real-time information was not retrieved rather than hallucinating.\n"
        f"5. Structured & Comprehensive: Present answers with clean markdown headings, bullet points, step-by-step logic, and copyable code blocks where appropriate.\n"
        f"6. Persona: Maintain Leo's cheerful, enthusiastic lion mascot identity (with occasional 🐾 or friendly smiles), ensuring educational excellence and clarity."
    )


# 2. Supported models configuration mirrored from app.py
ALL_MODELS: Dict[str, Dict[str, str]] = {
    "gemini-3-flash-preview": {
        "provider": "gemini",
        "name": "Gemini 3 Flash",
        "badge": "Google Gemini",
        "short_name": "Gemini 3 Flash"
    },
    "gemini-flash-latest": {
        "provider": "gemini",
        "name": "Gemini Flash Latest",
        "badge": "Google Gemini",
        "short_name": "Gemini Flash"
    },
    "gemini-flash-lite-latest": {
        "provider": "gemini",
        "name": "Gemini Flash Lite",
        "badge": "Google Gemini",
        "short_name": "Gemini Flash-Lite"
    },
    "gemini-3.5-flash": {
        "provider": "gemini",
        "name": "Gemini 3.5 Flash",
        "badge": "Google Gemini",
        "short_name": "Gemini 3.5 Flash"
    },
    "gemma-4-26b-a4b-it": {
        "provider": "gemini",
        "name": "Gemma 4 26B",
        "badge": "Google Gemini",
        "short_name": "Gemma 4"
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "name": "GPT-4o Mini",
        "badge": "OpenAI ChatGPT",
        "short_name": "ChatGPT 4o-Mini"
    },
    "gpt-4o": {
        "provider": "openai",
        "name": "GPT-4o",
        "badge": "OpenAI ChatGPT",
        "short_name": "ChatGPT 4o"
    },
    "claude-3-5-haiku-20241022": {
        "provider": "anthropic",
        "name": "Claude 3.5 Haiku",
        "badge": "Anthropic Claude",
        "short_name": "Claude 3.5 Haiku"
    },
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "name": "Claude 3.5 Sonnet",
        "badge": "Anthropic Claude",
        "short_name": "Claude 3.5 Sonnet"
    },
}


# 3. Environment & Key loader (Never logs or prints keys)
def load_active_keys() -> Dict[str, str]:
    """
    Loads active keys from .env.
    Guarantees secrets are never logged or printed.
    """
    # Look for .env in current directory or parent directory
    load_dotenv(override=True)
    if not os.getenv("GEMINI_API_KEY") and os.path.exists("../.env"):
        load_dotenv("../.env", override=True)

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    google_search_key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
    google_cse_id = os.getenv("GOOGLE_CSE_ID", "").strip()

    keys = {
        "gemini": gemini_key if gemini_key and "your_gemini" not in gemini_key else "",
        "openai": openai_key if openai_key and "your_openai" not in openai_key else "",
        "anthropic": anthropic_key if anthropic_key and "your_anthropic" not in anthropic_key else "",
        "google_search": google_search_key if google_search_key and "your_google_search" not in google_search_key else "",
        "google_cse": google_cse_id if google_cse_id and "your_google_cse" not in google_cse_id else "",
    }
    return keys


# 4. Rate-limit and Quota error detector (mirrored from app.py line 639-655)
def is_rate_limit_error(exc: Exception) -> bool:
    """Detects 429 rate limit or quota exhaustion across Gemini, OpenAI, and Anthropic."""
    if isinstance(exc, (exceptions.ResourceExhausted, exceptions.TooManyRequests)):
        return True
    try:
        if isinstance(exc, openai.RateLimitError):
            return True
    except Exception:
        pass
    try:
        if isinstance(exc, anthropic.RateLimitError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return any(
        term in msg for term in [
            "429", "quota", "resource_exhausted", "rate limit",
            "ratelimit", "tokens per minute", "requests per minute", "exceeded"
        ]
    )


# 5. Streaming Generator Adapter (mirrored from app.py line 682-755)
def start_streaming_provider(
    provider: str,
    model_name: str,
    api_key: str,
    prompt: str,
    history_msgs: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.2
) -> Generator[str, None, None]:
    """
    Initializes and yields tokens for the designated provider.
    Mirrors start_streaming_provider in app.py without Streamlit dependencies.
    """
    if history_msgs is None:
        history_msgs = []

    sys_prompt = get_leo_system_prompt()

    if provider == "gemini":
        genai.configure(api_key=api_key, transport="rest")
        gemini_history = []
        for msg in history_msgs:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        try:
            model_instance = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=sys_prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature, max_output_tokens=400)
            )
        except Exception:
            model_instance = genai.GenerativeModel(
                model_name=model_name,
                generation_config=genai.types.GenerationConfig(temperature=temperature, max_output_tokens=400)
            )
        chat_session = model_instance.start_chat(history=gemini_history)
        try:
            response_stream = chat_session.send_message(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except exceptions.BadRequest as br_err:
            if "stream=False" in str(br_err) or "400" in str(br_err):
                res = chat_session.send_message(prompt, stream=False)
                if res.text:
                    yield res.text
            else:
                raise br_err

    elif provider == "openai":
        client = openai.OpenAI(api_key=api_key)
        messages = [{"role": "system", "content": sys_prompt}]
        for msg in history_msgs:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        claude_messages = []
        for msg in history_msgs:
            role = "user" if msg["role"] == "user" else "assistant"
            if claude_messages and claude_messages[-1]["role"] == role:
                claude_messages[-1]["content"] += "\n\n" + msg["content"]
            else:
                claude_messages.append({"role": role, "content": msg["content"]})

        if not claude_messages or claude_messages[-1]["role"] != "user":
            claude_messages.append({"role": "user", "content": prompt})
        else:
            claude_messages[-1]["content"] += "\n\n" + prompt

        with client.messages.stream(
            model=model_name,
            system=sys_prompt,
            messages=claude_messages,
            max_tokens=2048,
            temperature=temperature
        ) as stream:
            for text in stream.text_stream:
                yield text
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# 6. Synchronous full answer collector with timing
def generate_full_response(
    provider: str,
    model_name: str,
    api_key: str,
    prompt: str,
    temperature: float = 0.2
) -> Tuple[str, float, float]:
    """
    Executes streaming generation and measures:
    - full_text: The complete generated response
    - ttft: Time to first token in seconds
    - total_time: Total elapsed generation time in seconds
    """
    t0 = time.time()
    stream = start_streaming_provider(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        prompt=prompt,
        temperature=temperature
    )

    first_chunk = next(stream)
    ttft = time.time() - t0

    tokens = [first_chunk]
    for chunk in stream:
        tokens.append(chunk)

    total_time = time.time() - t0
    full_text = "".join(tokens)
    return full_text, ttft, total_time


# 7. Candidate Fallback Engine (mirrored from app.py lines 1260-1400)
def execute_with_candidate_fallback(
    primary_model: str,
    active_keys: Dict[str, str],
    prompt: str,
    temperature: float = 0.2,
    mock_force_primary_429: bool = False
) -> Dict[str, Any]:
    """
    Executes a query following the exact candidate queue order from app.py:
    Gemini -> Alternate Gemini models -> OpenAI (gpt-4o-mini) -> Anthropic (claude-3-5-haiku).
    Supports mock_force_primary_429 for evaluation benchmarking.
    """
    primary_prov = ALL_MODELS[primary_model]["provider"]
    candidates: List[Tuple[str, str]] = [(primary_model, primary_prov)]

    # Build fallback candidates exactly as app.py does
    if primary_prov == "gemini":
        for backup_m in ["gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.5-flash"]:
            if backup_m != primary_model:
                candidates.append((backup_m, "gemini"))
        if active_keys.get("openai"):
            candidates.append(("gpt-4o-mini", "openai"))
        if active_keys.get("anthropic"):
            candidates.append(("claude-3-5-haiku-20241022", "anthropic"))
    elif primary_prov == "openai":
        for backup_m in ["gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest"]:
            candidates.append((backup_m, "gemini"))
        if active_keys.get("anthropic"):
            candidates.append(("claude-3-5-haiku-20241022", "anthropic"))
        if primary_model != "gpt-4o-mini":
            candidates.append(("gpt-4o-mini", "openai"))
    elif primary_prov == "anthropic":
        for backup_m in ["gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest"]:
            candidates.append((backup_m, "gemini"))
        if active_keys.get("openai"):
            candidates.append(("gpt-4o-mini", "openai"))

    # Filter to candidates that have active keys
    executable_candidates = []
    seen = set()
    for m_name, prov in candidates:
        if active_keys.get(prov) and (m_name, prov) not in seen:
            executable_candidates.append((m_name, prov))
            seen.add((m_name, prov))

    start_time = time.time()
    successful_model = None
    successful_provider = None
    full_response = ""
    failover_occurred = False
    attempts = []

    for attempt_idx, (cand_model, cand_prov) in enumerate(executable_candidates):
        cand_key = active_keys[cand_prov]
        try:
            # Check if we should mock a 429 for the primary model trial
            if mock_force_primary_429 and attempt_idx == 0:
                raise exceptions.ResourceExhausted("Mocked HTTP 429: ResourceExhausted for eval_failover")

            text, ttft, dur = generate_full_response(
                provider=cand_prov,
                model_name=cand_model,
                api_key=cand_key,
                prompt=prompt,
                temperature=temperature
            )
            full_response = text
            successful_model = cand_model
            successful_provider = cand_prov
            if attempt_idx > 0:
                failover_occurred = True
            attempts.append({"model": cand_model, "provider": cand_prov, "status": "success", "duration": dur})
            break

        except Exception as exc:
            is_rl = is_rate_limit_error(exc)
            attempts.append({
                "model": cand_model,
                "provider": cand_prov,
                "status": "failed",
                "is_rate_limit": is_rl,
                "error": str(exc)[:120]
            })
            continue

    total_elapsed = time.time() - start_time
    return {
        "success": successful_model is not None,
        "successful_model": successful_model,
        "successful_provider": successful_provider,
        "failover_occurred": failover_occurred,
        "total_elapsed": total_elapsed,
        "attempts": attempts,
        "response": full_response
    }
