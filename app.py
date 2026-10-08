"""
=============================================================================
Project: Leo 🦁 - AI Assistant Chatbot with Cartoon Mascot & Fast Streaming
Subject: MCA LLM Project
Tech Stack: Python 3.10+, Streamlit, Google Gemini API, OpenAI, Claude & Google Search API
=============================================================================
This application demonstrates:
1. Animated Cartoon AI Lion Mascot on screen with interactive speech bubble.
2. Multi-turn conversational memory with sliding token window for speed.
3. Fast word-by-word streaming using Google Gemini 2.0/3.0 Flash LLM.
4. Real-time Google Search & Live News Grounding (Custom Search API & News Feed).
5. Intelligent automatic fallback when Gemini free-tier rate limits are reached.
6. 1-Click retry and clear recovery guidance for 15 RPM / 1500 RPD quotas.
7. Robust exception handling for API keys, network issues, and quotas.
=============================================================================
"""

import os
import time
import base64
from datetime import datetime
import streamlit as st
import google.generativeai as genai
from google.api_core import exceptions
import openai
import anthropic
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv
import google_search

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CARTOON LION THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Leo 🦁 - Your Cartoon AI Lion Assistant",
    page_icon="🦁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Helper to load and cache base64 image strings for instant zero-latency rendering
@st.cache_data
def get_image_base64(filepath: str) -> str:
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

HERO_LION_B64 = get_image_base64("assets/cartoon_lion.jpg")
THUMB_LION_B64 = get_image_base64("assets/cartoon_lion_thumb.jpg")
LION_AVATAR_PATH = "assets/cartoon_lion_thumb.jpg" if os.path.exists("assets/cartoon_lion_thumb.jpg") else "🦁"

# Advanced 3D, Cartoon Mascot & Glassmorphic Custom Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Ambient background gradient illumination */
    .stApp {
        background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.14) 0%, rgba(15, 23, 42, 0) 65%),
                    radial-gradient(circle at 100% 40%, rgba(245, 158, 11, 0.1) 0%, rgba(15, 23, 42, 0) 50%),
                    radial-gradient(circle at 0% 80%, rgba(6, 182, 212, 0.08) 0%, rgba(15, 23, 42, 0) 50%);
    }

    /* 3D Hero Cartoon Lion Showcase */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 0.2rem 0 1.6rem 0;
        perspective: 1000px;
    }

    .hero-card-3d {
        position: relative;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-top: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 28px;
        padding: 1.4rem 1.8rem;
        width: 100%;
        box-shadow: 0 20px 45px -12px rgba(0, 0, 0, 0.55),
                    0 0 35px rgba(6, 182, 212, 0.15),
                    inset 0 1px 1px rgba(255, 255, 255, 0.25);
        transform-style: preserve-3d;
        transition: transform 0.4s ease, box-shadow 0.4s ease;
    }

    .hero-card-3d:hover {
        transform: translateY(-3px) scale(1.006);
        box-shadow: 0 25px 55px -10px rgba(0, 0, 0, 0.65),
                    0 0 45px rgba(6, 182, 212, 0.25);
    }

    /* Animated Cartoon Lion Mascot Frame */
    .cartoon-lion-frame {
        position: relative;
        flex-shrink: 0;
        width: 115px;
        height: 115px;
        border-radius: 24px;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 50%, #06b6d4 100%);
        padding: 3px;
        box-shadow: 0 10px 25px -4px rgba(245, 158, 11, 0.45),
                    0 0 25px rgba(6, 182, 212, 0.4);
        animation: floatLion 3.8s ease-in-out infinite;
    }

    .cartoon-lion-img {
        width: 100%;
        height: 100%;
        border-radius: 21px;
        object-fit: cover;
        display: block;
    }

    .cartoon-lion-badge {
        position: absolute;
        bottom: -6px;
        right: -6px;
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        color: #ffffff;
        font-family: 'Fredoka', cursive;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 2px 7px;
        border-radius: 9999px;
        border: 2px solid #0f172a;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    }

    @keyframes floatLion {
        0%, 100% {
            transform: translateY(0px) rotate(0deg);
        }
        50% {
            transform: translateY(-8px) rotate(1.5deg);
        }
    }

    /* Comic Speech Bubble next to Cartoon Lion */
    .speech-bubble {
        position: relative;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.92) 100%);
        border: 1px solid rgba(6, 182, 212, 0.35);
        border-radius: 18px;
        padding: 0.75rem 1.1rem;
        margin-top: 0.4rem;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.4),
                    0 0 15px rgba(6, 182, 212, 0.12);
    }

    .speech-bubble::before {
        content: '';
        position: absolute;
        left: -10px;
        top: 18px;
        width: 0;
        height: 0;
        border-top: 8px solid transparent;
        border-bottom: 8px solid transparent;
        border-right: 10px solid rgba(6, 182, 212, 0.4);
    }

    .speech-text {
        font-family: 'Fredoka', sans-serif;
        color: #ffffff !important;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.4;
    }

    .speech-highlight {
        color: #38bdf8 !important;
        font-weight: 700;
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.1;
        color: #ffffff !important;
    }

    .gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #a78bfa 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.4);
    }

    .hero-subtitle {
        color: #cbd5e1 !important;
        font-size: 0.92rem;
        margin: 0.2rem 0 0 0;
        font-weight: 500;
    }

    /* Live Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 0.25rem 0.85rem;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* 3D Elevated Chat Messages with High Contrast Visible Text */
    [data-testid="stChatMessage"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.92) 0%, rgba(15, 23, 42, 0.96) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.28) !important;
        border-radius: 20px !important;
        padding: 1.15rem 1.35rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5),
                    0 4px 6px -4px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(14px) !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease !important;
    }

    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.6) !important;
    }

    /* Force all text inside chat messages to be crisp, bright, and completely visible */
    [data-testid="stChatMessage"] * {
        color: #f8fafc !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] li {
        color: #f1f5f9 !important;
        font-size: 1rem !important;
        line-height: 1.65 !important;
    }

    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] b {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    [data-testid="stChatMessage"] code {
        background: rgba(15, 23, 42, 0.8) !important;
        color: #fbbf24 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
    }

    /* Chat Avatar Styling */
    [data-testid="stChatMessageAvatarCustom"], [data-testid="stChatMessageAvatar"] {
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
        border: 1.5px solid rgba(6, 182, 212, 0.45) !important;
    }

    /* Buttons */
    [data-testid="stButton"] button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 12px !important;
        padding: 0.55rem 1.1rem !important;
        box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.35),
                    inset 0 1px 1px rgba(255, 255, 255, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    [data-testid="stButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px 0 rgba(2, 132, 199, 0.55) !important;
        color: #ffffff !important;
    }

    [data-testid="stButton"] button p {
        color: #ffffff !important;
    }

    /* Rate Limit Notice Card */
    .rate-limit-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(245, 158, 11, 0.12) 100%);
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }

    .rate-limit-card * {
        color: #f1f5f9 !important;
    }

    /* Floating Screen Companion (Cartoon Lion in Corner) */
    .floating-companion-box {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 99999;
        display: flex;
        align-items: center;
        gap: 10px;
        pointer-events: auto;
        animation: floatLion 4.2s ease-in-out infinite;
    }

    .companion-avatar {
        width: 62px;
        height: 62px;
        border-radius: 18px;
        background: linear-gradient(135deg, #f59e0b, #06b6d4);
        padding: 2.5px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6), 0 0 20px rgba(6, 182, 212, 0.5);
        cursor: pointer;
        transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    .companion-avatar:hover {
        transform: scale(1.15) rotate(-3deg);
    }

    .companion-avatar img {
        width: 100%;
        height: 100%;
        border-radius: 15px;
        object-fit: cover;
        display: block;
    }

    .companion-tooltip {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(6, 182, 212, 0.4);
        border-radius: 12px;
        padding: 6px 12px;
        color: #f1f5f9;
        font-family: 'Fredoka', sans-serif;
        font-size: 0.8rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        white-space: nowrap;
    }

    /* Sleek Sidebar with Crystal Clear Text Visibility */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
    }

    /* Force all text in sidebar to be high contrast and clearly readable */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {
        color: #f8fafc !important;
        font-size: 0.93rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #94a3b8 !important;
    }

    /* Sidebar text input, password, select box styling */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] input::placeholder {
        color: #94a3b8 !important;
    }

    /* High contrast Alert / Warning boxes in sidebar */
    [data-testid="stAlert"] {
        background: rgba(30, 41, 59, 0.95) !important;
        border: 1px solid rgba(245, 158, 11, 0.6) !important;
        border-radius: 12px !important;
    }

    [data-testid="stAlert"] * {
        color: #fef08a !important;
    }

    /* Slider styling in sidebar */
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background-color: #06b6d4 !important;
        border-color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] div {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 2. ENVIRONMENT & API KEY INITIALIZATION
# -----------------------------------------------------------------------------
load_dotenv(override=True)
env_gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
env_openai_key = os.getenv("OPENAI_API_KEY", "").strip()
env_anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
env_google_search_key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
env_google_cse_id = os.getenv("GOOGLE_CSE_ID", "").strip()

is_gemini_placeholder = env_gemini_key in ("", "your_gemini_api_key_here")
is_openai_placeholder = env_openai_key in ("", "your_openai_api_key_here")
is_anthropic_placeholder = env_anthropic_key in ("", "your_anthropic_api_key_here")
is_google_search_placeholder = env_google_search_key in ("", "your_google_search_api_key_here")
is_google_cse_placeholder = env_google_cse_id in ("", "your_google_cse_id_here")

# Default active keys
active_gemini_key = env_gemini_key if not is_gemini_placeholder else ""
active_openai_key = env_openai_key if not is_openai_placeholder else ""
active_anthropic_key = env_anthropic_key if not is_anthropic_placeholder else ""
active_google_search_key = env_google_search_key if not is_google_search_placeholder else ""
active_google_cse_id = env_google_cse_id if not is_google_cse_placeholder else ""

# -----------------------------------------------------------------------------
# 3. HIGH-ACCURACY SYSTEM PROMPT & LIVE KNOWLEDGE ANCHOR
# -----------------------------------------------------------------------------
def get_leo_system_prompt() -> str:
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

LEO_SYSTEM_PROMPT = get_leo_system_prompt()


def fetch_live_knowledge_snippet(query: str) -> str:
    """Fetches real-time encyclopedic facts from Wikipedia and DuckDuckGo to ground responses."""
    snippets = []
    clean_q = query.strip().rstrip("?").rstrip(".").strip()
    # 1. Wikipedia summary
    try:
        wiki_q = clean_q.replace(" ", "_")
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(wiki_q)}"
        req = urllib.request.Request(wiki_url, headers={"User-Agent": "LeoAIChatbot/1.0 (Educational)"})
        with urllib.request.urlopen(req, timeout=2.5) as r:
            data = json.loads(r.read().decode("utf-8"))
            extract = data.get("extract")
            if extract and len(extract) > 40:
                snippets.append(f"[Wikipedia Fact Sheet on {clean_q}]:\n{extract}")
    except Exception:
        pass

    # 2. DuckDuckGo Instant Answer
    try:
        ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_q)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(ddg_url, headers={"User-Agent": "LeoAIChatbot/1.0"})
        with urllib.request.urlopen(req, timeout=2.5) as r:
            data = json.loads(r.read().decode("utf-8"))
            abstract = data.get("AbstractText", "") or data.get("Abstract", "")
            if abstract and len(abstract) > 30 and abstract not in "\n".join(snippets):
                snippets.append(f"[Verified Instant Fact]:\n{abstract}")
    except Exception:
        pass

    return "\n\n".join(snippets)

# Supported models across providers (active 2026 model names)
ALL_MODELS = {
    # Google Gemini Models (Active endpoints with fresh separate free quotas)
    "gemini-3-flash-preview": {
        "provider": "gemini",
        "name": "Gemini 3 Flash (Google - Fresh Quota & High Accuracy 🧠)",
        "badge": "Google Gemini",
        "short_name": "Gemini 3 Flash"
    },
    "gemini-flash-latest": {
        "provider": "gemini",
        "name": "Gemini Flash Latest (Google - Stable High-Capacity ⚡)",
        "badge": "Google Gemini",
        "short_name": "Gemini Flash"
    },
    "gemini-flash-lite-latest": {
        "provider": "gemini",
        "name": "Gemini Flash Lite (Google - Ultra-Fast Fallback 🛡️)",
        "badge": "Google Gemini",
        "short_name": "Gemini Flash-Lite"
    },
    "gemini-3.5-flash": {
        "provider": "gemini",
        "name": "Gemini 3.5 Flash (Google - Standard Flash 🌟)",
        "badge": "Google Gemini",
        "short_name": "Gemini 3.5 Flash"
    },
    # OpenAI ChatGPT Models
    "gpt-4o-mini": {
        "provider": "openai",
        "name": "GPT-4o Mini (ChatGPT / OpenAI - Fast & Smart 🟢)",
        "badge": "OpenAI ChatGPT",
        "short_name": "ChatGPT 4o-Mini"
    },
    "gpt-4o": {
        "provider": "openai",
        "name": "GPT-4o (ChatGPT / OpenAI - Flagship Omni 🌟)",
        "badge": "OpenAI ChatGPT",
        "short_name": "ChatGPT 4o"
    },
    # Anthropic Claude Models
    "claude-3-5-haiku-20241022": {
        "provider": "anthropic",
        "name": "Claude 3.5 Haiku (Anthropic - Lightning Speed 🟣)",
        "badge": "Anthropic Claude",
        "short_name": "Claude 3.5 Haiku"
    },
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "name": "Claude 3.5 Sonnet (Anthropic - State-of-the-Art 🎯)",
        "badge": "Anthropic Claude",
        "short_name": "Claude 3.5 Sonnet"
    },
}

# -----------------------------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_retry_prompt" not in st.session_state:
    st.session_state.pending_retry_prompt = None

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = None

if "show_floating_lion" not in st.session_state:
    st.session_state.show_floating_lion = True

# -----------------------------------------------------------------------------
# 5. MULTI-PROVIDER HELPER UTILITIES
# -----------------------------------------------------------------------------
def save_all_keys_to_env(gemini_k: str, openai_k: str, anthropic_k: str, google_search_k: str = "", google_cse_k: str = ""):
    """Safely updates or persists API keys in the .env file without deleting existing entries."""
    existing_env = {}
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str and not line_str.startswith("#") and "=" in line_str:
                    k, v = line_str.split("=", 1)
                    existing_env[k.strip()] = v.strip()

    if gemini_k:
        existing_env["GEMINI_API_KEY"] = gemini_k
    if openai_k:
        existing_env["OPENAI_API_KEY"] = openai_k
    if anthropic_k:
        existing_env["ANTHROPIC_API_KEY"] = anthropic_k
    if google_search_k is not None:
        existing_env["GOOGLE_SEARCH_API_KEY"] = google_search_k
    if google_cse_k is not None:
        existing_env["GOOGLE_CSE_ID"] = google_cse_k

    with open(".env", "w", encoding="utf-8") as f:
        f.write("# API Keys Configuration for Leo AI Chatbot\n")
        for k, v in existing_env.items():
            f.write(f"{k}={v}\n")


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
    return any(term in msg for term in ["429", "quota", "resource_exhausted", "rate limit", "ratelimit", "tokens per minute", "requests per minute", "exceeded"])


def test_provider_connection(provider: str, model_name: str, key: str):
    """Runs a minimal ping to verify API key validity, latency, and quota."""
    t0 = time.time()
    if provider == "gemini":
        genai.configure(api_key=key, transport="rest")
        m = genai.GenerativeModel(model_name)
        m.generate_content("Reply with 'OK'", stream=False)
    elif provider == "openai":
        client = openai.OpenAI(api_key=key)
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with 'OK'"}],
            max_tokens=5
        )
    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=key)
        client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with 'OK'"}],
            max_tokens=5
        )
    latency = round((time.time() - t0) * 1000)
    return latency


def start_streaming_provider(provider: str, model_name: str, api_key: str, prompt: str, history_msgs: list, temperature: float):
    """Initializes and yields tokens for the designated provider."""
    sys_prompt = get_leo_system_prompt()

    if provider == "gemini":
        genai.configure(api_key=api_key, transport="rest")
        gemini_history = []
        for msg in history_msgs:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
        model_instance = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=sys_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature
            )
        )
        chat_session = model_instance.start_chat(history=gemini_history)
        try:
            response_stream = chat_session.send_message(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except exceptions.BadRequest as br_err:
            # Handle REST streaming 400 error by falling back to direct send_message
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


# -----------------------------------------------------------------------------
# 6. SIDEBAR CONTROLS & MULTI-PROVIDER KEY MANAGER
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🦁 Leo's Control Center")
    st.caption("MCA LLM Project &bull; Multi-Provider AI Companion")
    st.markdown("---")

    st.markdown("#### 🔑 Multi-Provider API Keys")
    st.caption("Configure keys below. When one provider hits a rate limit, Leo automatically cascades to the next!")

    # Tabs for the providers including Google Search
    tab_gemini, tab_openai, tab_claude, tab_search = st.tabs(["🔮 Gemini", "🟢 ChatGPT", "🟣 Claude", "🔍 Google Search"])

    with tab_gemini:
        gemini_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy... (free tier)",
            help="Get a free key with generous quota at https://aistudio.google.com/app/apikey",
            key="input_gemini_key"
        )
        if gemini_input.strip():
            active_gemini_key = gemini_input.strip()
            st.success("Session Key Active", icon="⚡")
        elif not is_gemini_placeholder:
            active_gemini_key = env_gemini_key
            st.success("Key Active (.env)", icon="🔑")
        else:
            active_gemini_key = ""
            st.warning("Not configured", icon="⚪")

    with tab_openai:
        openai_input = st.text_input(
            "OpenAI API Key (ChatGPT)",
            type="password",
            placeholder="sk-... (for GPT-4o / GPT-4o Mini)",
            help="Get your OpenAI key at https://platform.openai.com/api-keys",
            key="input_openai_key"
        )
        if openai_input.strip():
            active_openai_key = openai_input.strip()
            st.success("Session Key Active", icon="⚡")
        elif not is_openai_placeholder:
            active_openai_key = env_openai_key
            st.success("Key Active (.env)", icon="🔑")
        else:
            active_openai_key = ""
            st.info("Not configured (Optional Fallback)", icon="⚪")

    with tab_claude:
        anthropic_input = st.text_input(
            "Anthropic API Key (Claude)",
            type="password",
            placeholder="sk-ant-... (for Claude 3.5 Haiku/Sonnet)",
            help="Get your Anthropic key at https://console.anthropic.com/settings/keys",
            key="input_anthropic_key"
        )
        if anthropic_input.strip():
            active_anthropic_key = anthropic_input.strip()
            st.success("Session Key Active", icon="⚡")
        elif not is_anthropic_placeholder:
            active_anthropic_key = env_anthropic_key
            st.success("Key Active (.env)", icon="🔑")
        else:
            active_anthropic_key = ""
            st.info("Not configured (Optional Fallback)", icon="⚪")

    with tab_search:
        gsearch_input = st.text_input(
            "Google Search API Key",
            type="password",
            placeholder="AIzaSy... (Google Cloud API Key)",
            help="Get your key at https://developers.google.com/custom-search/v1/overview",
            key="input_google_search_key"
        )
        cse_input = st.text_input(
            "Search Engine ID (CX)",
            placeholder="e.g. 0175766625124... (or cx string)",
            help="Create a Programmable Search Engine at https://programmablesearchengine.google.com/",
            key="input_google_cse_id"
        )
        if gsearch_input.strip() and cse_input.strip():
            active_google_search_key = gsearch_input.strip()
            active_google_cse_id = cse_input.strip()
            st.success("Official Custom Search API Active", icon="⚡")
        elif not is_google_search_placeholder and not is_google_cse_placeholder and env_google_search_key and env_google_cse_id:
            active_google_search_key = env_google_search_key
            active_google_cse_id = env_google_cse_id
            st.success("Custom Search Active (.env)", icon="🔑")
        else:
            active_google_search_key = gsearch_input.strip() or env_google_search_key
            active_google_cse_id = cse_input.strip() or env_google_cse_id
            st.info("📰 Google News Live Search Ready (Free Zero-Key Active)", icon="⚡")

        with st.expander("ℹ️ How to get Google Search API Credentials", expanded=False):
            st.markdown(
                """
                **Official Google Custom Search JSON API:**
                1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/library/customsearch.googleapis.com) and enable **Custom Search API**.
                2. In **Credentials**, create an **API Key**.
                3. Go to [cse.google.com](https://programmablesearchengine.google.com/), click **Add**, choose **Search the entire web**, and copy your **Search Engine ID (CX)**.
                
                *⚡ Real-time Google News search works out-of-the-box even without an API key!*
                """
            )

    # Map of active keys
    active_keys = {
        "gemini": active_gemini_key,
        "openai": active_openai_key,
        "anthropic": active_anthropic_key
    }

    # Save and Test Buttons
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if st.button("💾 Save Keys", use_container_width=True, help="Save all active keys permanently to .env file"):
            save_all_keys_to_env(
                active_gemini_key,
                active_openai_key,
                active_anthropic_key,
                active_google_search_key,
                active_google_cse_id
            )
            st.toast("✅ Keys saved permanently to .env!")
            time.sleep(0.3)
            st.rerun()

    with col_k2:
        test_clicked = st.button("🧪 Test Keys", use_container_width=True, help="Test connectivity and latency for all configured keys")

    if test_clicked:
        configured_any = False
        with st.spinner("Pinging configured AI providers & Google Search..."):
            if active_gemini_key:
                configured_any = True
                try:
                    lat = test_provider_connection("gemini", "gemini-3.5-flash", active_gemini_key)
                    st.success(f"🔮 **Gemini**: Online ({lat}ms)", icon="⚡")
                except Exception as e:
                    if is_rate_limit_error(e):
                        st.warning("🔮 **Gemini**: Rate limited / quota wait", icon="⏳")
                    else:
                        st.error(f"🔮 **Gemini**: {str(e)[:90]}", icon="❌")

            if active_openai_key:
                configured_any = True
                try:
                    lat = test_provider_connection("openai", "gpt-4o-mini", active_openai_key)
                    st.success(f"🟢 **ChatGPT**: Online ({lat}ms)", icon="⚡")
                except Exception as e:
                    err_str = str(e).lower()
                    if "credit_balance_exhausted" in err_str or "insufficient_quota" in err_str:
                        st.error("🟢 **ChatGPT**: 0 Credits Remaining (Check OpenAI billing)", icon="💳")
                    elif is_rate_limit_error(e):
                        st.warning("🟢 **ChatGPT**: Rate limited / quota wait", icon="⏳")
                    else:
                        st.error(f"🟢 **ChatGPT**: {str(e)[:90]}", icon="❌")

            if active_anthropic_key:
                configured_any = True
                try:
                    lat = test_provider_connection("anthropic", "claude-3-5-haiku-20241022", active_anthropic_key)
                    st.success(f"🟣 **Claude**: Online ({lat}ms)", icon="⚡")
                except Exception as e:
                    if is_rate_limit_error(e):
                        st.warning("🟣 **Claude**: Rate limited / quota wait", icon="⏳")
                    else:
                        st.error(f"🟣 **Claude**: {str(e)[:90]}", icon="❌")

            # Test Google Search Engine connectivity
            try:
                g_lat, g_engine = google_search.test_google_search_connection(
                    active_google_search_key, active_google_cse_id
                )
                if active_google_search_key and active_google_cse_id:
                    st.success(f"🔍 **Google Search API**: Online ({g_lat}ms) &bull; Custom Search", icon="⚡")
                else:
                    st.success(f"📰 **Google News Live Search**: Online ({g_lat}ms) &bull; Real-Time Active", icon="⚡")
            except Exception as e:
                st.warning(f"🔍 **Google Search**: {str(e)[:90]}", icon="⚠️")

            if not configured_any:
                st.warning("Please enter at least one API key above.", icon="⚠️")

    st.markdown("---")
    st.markdown("#### ⚙️ Speed & Model Settings")

    # Model selector
    model_choice = st.selectbox(
        label="Select Primary AI Model",
        options=list(ALL_MODELS.keys()),
        format_func=lambda x: ALL_MODELS[x]["name"],
        index=0,
        help="Leo will start with this model. If rate limits or quota errors occur, Leo seamlessly cascades to backup models."
    )

    primary_provider = ALL_MODELS[model_choice]["provider"]
    has_primary_key = bool(active_keys.get(primary_provider))

    # Multi-provider Auto-Fallback toggle
    enable_auto_fallback = st.checkbox(
        "🛡️ Auto-Fallback on Rate Limit",
        value=True,
        help="Automatically switch to ChatGPT, Claude, or alternate Gemini models if primary model hits 429 quota limits."
    )

    # Calculate and display current fallback route
    fallback_route = [ALL_MODELS[model_choice]["short_name"]]
    if enable_auto_fallback:
        if primary_provider == "gemini":
            for backup_m in ["gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.5-flash"]:
                if backup_m != model_choice:
                    fallback_route.append(ALL_MODELS[backup_m]["short_name"])
            if active_keys.get("openai"):
                fallback_route.append("ChatGPT 4o-Mini")
            if active_keys.get("anthropic"):
                fallback_route.append("Claude 3.5 Haiku")
        elif primary_provider == "openai":
            for backup_m in ["gemini-3-flash-preview", "gemini-flash-latest"]:
                fallback_route.append(ALL_MODELS[backup_m]["short_name"])
            if active_keys.get("anthropic"):
                fallback_route.append("Claude 3.5 Haiku")
            if model_choice != "gpt-4o-mini":
                fallback_route.append("GPT-4o Mini")
        elif primary_provider == "anthropic":
            for backup_m in ["gemini-3-flash-preview", "gemini-flash-latest"]:
                fallback_route.append(ALL_MODELS[backup_m]["short_name"])
            if active_keys.get("openai"):
                fallback_route.append("ChatGPT 4o-Mini")

    st.caption(f"**Failover Route:** {' ➔ '.join(fallback_route)}")

    # Accuracy Mode Preset & Temperature
    response_mode = st.radio(
        label="🎯 Accuracy & Precision Mode",
        options=["🎯 High Accuracy (Factual & Coding)", "⚖️ Balanced", "🎨 Creative"],
        index=0,
        help="High Accuracy sets a low temperature (0.2) to eliminate hallucinations and give strictly factual, truthful answers."
    )
    suggested_temp = 0.2 if "High Accuracy" in response_mode else (0.5 if "Balanced" in response_mode else 0.8)

    temperature = st.slider(
        label="Fine-tune Temperature",
        min_value=0.0,
        max_value=1.0,
        value=suggested_temp,
        step=0.05,
        help="0.0 - 0.3 = strictly factual/coding/math. 0.7 - 1.0 = creative/speculative."
    )

    st.markdown("---")
    st.markdown("#### 🔍 Real-Time Google Search Grounding")

    # Real-Time Google Search Grounding toggle
    enable_google_search = st.toggle(
        "🌐 Google Search & Live News",
        value=True,
        help="Queries Google Custom Search API and live Google News in real time to fetch breaking news, current events, and live 2026 data."
    )

    if enable_google_search:
        search_trigger_mode = st.selectbox(
            "Search Trigger Mode",
            options=[
                "Smart Auto (Detects latest news, dates, and facts 🧠)",
                "Always Search Google (Every query 🔎)",
                "News Only (Headlines & breaking stories 📰)"
            ],
            index=0,
            help="Smart Auto detects when user is asking about current events, scores, breaking news, or recent developments."
        )
        search_results_count = st.slider(
            "Search Depth (Sources)",
            min_value=3,
            max_value=8,
            value=5,
            step=1,
            help="Number of Google search results and news items to retrieve."
        )
    else:
        search_trigger_mode = "Disabled"
        search_results_count = 5

    # Live Knowledge Grounding toggle (Wikipedia)
    enable_live_grounding = st.toggle(
        "📚 Wikipedia Encyclopedia Grounding",
        value=True,
        help="Retrieves live encyclopedic fact sheets to verify entity definitions, history, and scientific concepts."
    )

    # Sliding Memory Window for speed & quota protection
    history_window = st.slider(
        label="Memory Window (Past Messages)",
        min_value=2,
        max_value=16,
        value=8,
        step=2,
        help="Limit previous turns sent to the LLM. Keeps responses fast and avoids reaching token limits."
    )

    # Floating Mascot toggle
    st.session_state.show_floating_lion = st.toggle(
        "Show Floating Cartoon Lion on Screen",
        value=st.session_state.show_floating_lion,
        help="Displays the animated cartoon AI lion mascot on screen."
    )

    st.markdown("---")
    st.markdown("#### 💬 Chat Management")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_retry_prompt = None
        st.rerun()

    # Export chat history
    def format_chat_as_text(messages):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 50,
            "Leo 🦁 - AI Assistant Chat History",
            f"Exported At: {timestamp}",
            "=" * 50,
            ""
        ]
        for msg in messages:
            sender = "You" if msg["role"] == "user" else "Leo 🦁"
            lines.append(f"[{sender}]:\n{msg['content']}\n")
        return "\n".join(lines)

    st.download_button(
        label="📥 Download Conversation (.txt)",
        data=format_chat_as_text(st.session_state.messages),
        file_name=f"leo_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        disabled=len(st.session_state.messages) == 0,
        use_container_width=True
    )

    # Free Tier Quota Info
    st.markdown("---")
    st.markdown("#### ℹ️ Quota & Fallback Info")
    st.caption(
        "If Google Gemini hits rate limits (15 requests/min), Leo's **Multi-Provider Shield** "
        "automatically routes requests to ChatGPT or Claude to deliver zero-downtime chats!"
    )

# -----------------------------------------------------------------------------
# 7. MAIN CHAT INTERFACE: 3D CARTOON LION HERO SECTION
# -----------------------------------------------------------------------------
active_provider_label = ALL_MODELS[model_choice]["badge"]
cartoon_lion_html = f"""
<div class="hero-container">
    <div class="status-badge">
        <span class="status-dot"></span>
        <span>SYSTEM ONLINE &bull; {model_choice.upper()} &bull; MULTI-PROVIDER FAILOVER SHIELD</span>
    </div>
    <div class="hero-card-3d">
        <div class="cartoon-lion-frame">
            {"<img src='data:image/jpeg;base64," + HERO_LION_B64 + "' class='cartoon-lion-img' alt='Cartoon AI Lion' />" if HERO_LION_B64 else "<div style='font-size:3.5rem; text-align:center;'>🦁</div>"}
            <span class="cartoon-lion-badge">AI LION</span>
        </div>
        <div style="flex-grow: 1;">
            <h1 class="hero-title">LEO <span class="gradient-text">THE AI LION</span></h1>
            <p class="hero-subtitle">MCA LLM Project &bull; Cartoon Mascot &bull; Gemini, ChatGPT & Claude Fallback Shield</p>
            <div class="speech-bubble">
                <p class="speech-text">
                    🐾 <strong>Leo:</strong> "Rawr! Welcome, friend! I'm equipped with <span class="speech-highlight">{ALL_MODELS[model_choice]['short_name']}</span> and automatic failover to ChatGPT & Claude. What shall we explore today?"
                </p>
            </div>
        </div>
    </div>
</div>
"""
st.markdown(cartoon_lion_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. OPTIONAL FLOATING CARTOON LION COMPANION
# -----------------------------------------------------------------------------
if st.session_state.show_floating_lion and THUMB_LION_B64:
    floating_lion_html = f"""
    <div class="floating-companion-box">
        <div class="companion-tooltip">🐾 Leo is ready!</div>
        <div class="companion-avatar" title="Leo the Cartoon AI Lion">
            <img src="data:image/jpeg;base64,{THUMB_LION_B64}" alt="Leo Companion" />
        </div>
    </div>
    """
    st.markdown(floating_lion_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. DISPLAY EXISTING CONVERSATION HISTORY
# -----------------------------------------------------------------------------
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar=LION_AVATAR_PATH):
            s_data = message.get("search_data")
            if s_data and s_data.get("results"):
                with st.expander(
                    f"🔍 **Google Search Grounding:** {s_data.get('source_title', 'Google Search')} ({len(s_data['results'])} sources for *\"{s_data.get('query')}\"*)",
                    expanded=False
                ):
                    for idx, res in enumerate(s_data["results"], 1):
                        pub_info = f" &bull; *{res['published']}*" if res.get("published") else ""
                        source_info = f" &bull; **{res['source']}**" if res.get("source") else ""
                        st.markdown(f"**{idx}. [{res['title']}]({res['url']})**{source_info}{pub_info}")
                        if res.get("snippet"):
                            st.caption(f"> {res['snippet']}")
                        st.markdown("---")
            st.markdown(message["content"])

# Quick Starters when chat is empty
active_prompt = None
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant", avatar=LION_AVATAR_PATH):
        st.markdown(
            f"Greetings! I am **Leo**, your friendly cartoon AI lion! 🦁⚡ "
            f"I'm powered by **{ALL_MODELS[model_choice]['name']}** with multi-turn memory, "
            f"**Real-Time Google Search Grounding 🔍**, and multi-provider failover. "
            "Click a starter below or ask me anything!"
        )

    st.markdown(
        "<div style='margin-top: 1.2rem; margin-bottom: 0.5rem; color: #38bdf8; font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;'>⚡ Try a Quick Starter</div>",
        unsafe_allow_html=True
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📰 Latest AI News", use_container_width=True):
            active_prompt = "What is the latest breaking news and developments in AI and technology today?"
    with col2:
        if st.button("🧠 Self-Attention", use_container_width=True):
            active_prompt = "Can you clearly explain the Self-Attention mechanism in Transformer LLMs with an intuitive example?"
    with col3:
        if st.button("🚀 Fast Python Async", use_container_width=True):
            active_prompt = "Provide a clean Python script demonstrating asyncio concurrent tasks with error handling."
    with col4:
        if st.button("🦁 Who is Leo?", use_container_width=True):
            active_prompt = "Who are you, what models power you, and how does your Google Search grounding work?"

# -----------------------------------------------------------------------------
# 10. HANDLE 1-CLICK RETRY IF PREVIOUS REQUEST HIT RATE LIMIT
# -----------------------------------------------------------------------------
if st.session_state.pending_retry_prompt:
    st.markdown(
        """
        <div class="rate-limit-card">
            <h4 style="margin: 0 0 0.4rem 0; color: #f59e0b;">⏳ Rate Limit Recovery</h4>
            <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">
                Rate limit was reached on previous model. You can retry your question now or allow Leo to route via ChatGPT/Claude!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    col_ret1, col_ret2 = st.columns([2, 1])
    with col_ret1:
        if st.button(f"🔄 Retry Question: \"{st.session_state.pending_retry_prompt[:45]}...\"", use_container_width=True):
            active_prompt = st.session_state.pending_retry_prompt
            st.session_state.pending_retry_prompt = None
    with col_ret2:
        if st.button("❌ Dismiss Retry", use_container_width=True):
            st.session_state.pending_retry_prompt = None
            st.rerun()

# -----------------------------------------------------------------------------
# 11. HANDLE USER INPUT & GENERATE FAST STREAMING WITH FAILOVER
# -----------------------------------------------------------------------------
user_input_box = st.chat_input("Ask Leo anything...")
user_prompt = active_prompt if active_prompt else user_input_box

if user_prompt:
    # Check if ANY key is available
    has_any_key = any(bool(k) for k in active_keys.values())

    if not has_any_key:
        st.error(
            "⚠️ **API Key Required!**\n\n"
            "Please enter an API key in the sidebar under **Multi-Provider API Keys** (Google Gemini, OpenAI ChatGPT, or Claude) to talk to Leo.\n"
            "• Get free Gemini key: [Google AI Studio](https://aistudio.google.com/app/apikey)\n"
            "• Get OpenAI key: [OpenAI Platform](https://platform.openai.com/api-keys)\n"
            "• Get Anthropic key: [Anthropic Console](https://console.anthropic.com/settings/keys)",
            icon="🚨"
        )
    else:
        # Display user question
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_prompt)

        # Append to session history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Prepare trimmed conversation history for speed and quota preservation
        recent_messages = st.session_state.messages[:-1]
        if len(recent_messages) > history_window:
            recent_messages = recent_messages[-history_window:]

        # Build candidate models queue based on availability and fallback settings
        candidates = [(model_choice, primary_provider)]

        if enable_auto_fallback:
            if primary_provider == "gemini":
                # Immediately cascade through alternate Gemini models with separate quotas
                for backup_m in ["gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.5-flash"]:
                    if backup_m != model_choice:
                        candidates.append((backup_m, "gemini"))
                if active_keys.get("openai"):
                    candidates.append(("gpt-4o-mini", "openai"))
                if active_keys.get("anthropic"):
                    candidates.append(("claude-3-5-haiku-20241022", "anthropic"))
            elif primary_provider == "openai":
                for backup_m in ["gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest"]:
                    candidates.append((backup_m, "gemini"))
                if active_keys.get("anthropic"):
                    candidates.append(("claude-3-5-haiku-20241022", "anthropic"))
                if model_choice != "gpt-4o-mini":
                    candidates.append(("gpt-4o-mini", "openai"))
            elif primary_provider == "anthropic":
                for backup_m in ["gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest"]:
                    candidates.append((backup_m, "gemini"))
                if active_keys.get("openai"):
                    candidates.append(("gpt-4o-mini", "openai"))

        # Filter candidates: keep only those whose provider has an active key
        executable_candidates = []
        seen = set()
        for m_name, prov in candidates:
            if active_keys.get(prov) and (m_name, prov) not in seen:
                executable_candidates.append((m_name, prov))
                seen.add((m_name, prov))

        # Generate response
        with st.chat_message("assistant", avatar=LION_AVATAR_PATH):
            successful_stream = None
            successful_model = None
            successful_provider = None
            failover_occurred = False
            failed_attempts = []
            start_time = time.time()

            live_context = ""
            google_search_data = None

            # 1. Real-Time Google Search & News Grounding
            if enable_google_search:
                should_run_search = (
                    "Always Search" in search_trigger_mode
                    or google_search.should_search_google(user_prompt)
                )
                if should_run_search:
                    search_mode_param = "news" if "News Only" in search_trigger_mode else "auto"
                    with st.spinner("🔍 Leo is querying Google Search for live data & news..."):
                        google_search_data = google_search.execute_google_search(
                            query=user_prompt,
                            api_key=active_google_search_key,
                            cse_id=active_google_cse_id,
                            num_results=search_results_count,
                            mode=search_mode_param
                        )
                    if google_search_data and google_search_data.get("results"):
                        live_context += google_search_data.get("prompt_text", "")

            # 2. Encyclopedic Fact Sheet Grounding (Wikipedia)
            if enable_live_grounding:
                wiki_snippet = fetch_live_knowledge_snippet(user_prompt)
                if wiki_snippet:
                    live_context += ("\n\n" + wiki_snippet) if live_context else wiki_snippet

            effective_prompt = (
                f"[LIVE VERIFIED DATA & REAL-TIME GOOGLE SEARCH]\n{live_context}\n\n"
                f"[USER QUESTION]\n{user_prompt}\n\n"
                f"Please answer the user's question accurately using the live Google search results and verified data above when applicable. If referencing recent news or sources, mention the source name and publication date."
            ) if live_context else user_prompt

            # Render Google Search citation expander if live results were retrieved
            if google_search_data and google_search_data.get("results"):
                with st.expander(
                    f"🔍 **Google Search Grounding:** {google_search_data.get('source_title', 'Google Search')} ({len(google_search_data['results'])} sources for *\"{google_search_data.get('query')}\"*)",
                    expanded=False
                ):
                    for idx, res in enumerate(google_search_data["results"], 1):
                        pub_info = f" &bull; *{res['published']}*" if res.get("published") else ""
                        source_info = f" &bull; **{res['source']}**" if res.get("source") else ""
                        st.markdown(f"**{idx}. [{res['title']}]({res['url']})**{source_info}{pub_info}")
                        if res.get("snippet"):
                            st.caption(f"> {res['snippet']}")
                        st.markdown("---")
            elif live_context:
                st.caption("🌐 *Live Encyclopedia Grounding active for this query*")

            for attempt_idx, (cand_model, cand_provider) in enumerate(executable_candidates):
                cand_key = active_keys[cand_provider]
                try:
                    raw_gen = start_streaming_provider(
                        provider=cand_provider,
                        model_name=cand_model,
                        api_key=cand_key,
                        prompt=effective_prompt,
                        history_msgs=recent_messages,
                        temperature=temperature
                    )
                    # Pull first chunk to verify stream initiation & catch 429/404/quota immediately
                    first_token = next(raw_gen)

                    # Successful stream established!
                    def unified_generator(first_item, generator):
                        yield first_item
                        try:
                            for item in generator:
                                yield item
                        except Exception as mid_stream_err:
                            if is_rate_limit_error(mid_stream_err):
                                yield "\n\n*(Rate limit cooldown reached mid-stream. Next answer will auto-switch.)*"
                            else:
                                yield f"\n\n*(Stream interrupted: {mid_stream_err})*"

                    successful_stream = unified_generator(first_token, raw_gen)
                    successful_model = cand_model
                    successful_provider = cand_provider
                    if attempt_idx > 0:
                        failover_occurred = True

                    break

                except StopIteration:
                    # Model returned empty response, try next
                    failed_attempts.append({"model": cand_model, "error": "Empty response"})
                    continue
                except Exception as cand_err:
                    is_rl = is_rate_limit_error(cand_err)
                    failed_attempts.append({
                        "model": cand_model,
                        "provider": cand_provider,
                        "error": str(cand_err),
                        "is_rate_limit": is_rl
                    })
                    # If this was not the last candidate, continue to next
                    continue

            # Render response or error card
            if successful_stream is not None:
                # If failover was triggered, inform the user gracefully
                if failover_occurred:
                    st.info(
                        f"🔄 **Smart Multi-Provider Failover Triggered!**\n\n"
                        f"Primary model `{model_choice}` encountered a rate limit or service error. "
                        f"Leo automatically routed your question to **{ALL_MODELS[successful_model]['badge']} ({successful_model})** "
                        f"so your conversation didn't miss a beat! 🦁🛡️",
                        icon="⚡"
                    )

                try:
                    full_response = st.write_stream(successful_stream)
                    elapsed = time.time() - start_time
                    st.session_state.last_response_time = elapsed

                    # Persist response in session history with search citation data
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "search_data": google_search_data
                    })
                    st.session_state.pending_retry_prompt = None

                except Exception as write_err:
                    st.error(f"❌ Output display error: {str(write_err)}")
                    st.session_state.pending_retry_prompt = user_prompt
                    st.session_state.messages.pop()

            else:
                # All candidates failed
                st.session_state.pending_retry_prompt = user_prompt
                st.session_state.messages.pop()

                error_summary_lines = ""
                for fa in failed_attempts:
                    prov_badge = ALL_MODELS.get(fa['model'], {}).get('badge', fa.get('provider', 'AI'))
                    err_clean = fa['error'].replace('\n', ' ')[:120]
                    error_summary_lines += f"<li><strong>{prov_badge} ({fa['model']}):</strong> {err_clean}</li>"

                st.markdown(
                    f"""
                    <div class="rate-limit-card">
                        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                            <span style="font-size: 2rem;">🦁💤</span>
                            <div>
                                <strong style="color: #f59e0b; font-size: 1.05rem;">All Active Models Rate Limited or Unavailable</strong>
                                <p style="margin: 0; font-size: 0.88rem; color: #94a3b8;">
                                    Every configured API model hit rate limits or connection bounds.
                                </p>
                            </div>
                        </div>
                        <ul style="color: #cbd5e1; font-size: 0.84rem; margin: 0.5rem 0 0.8rem 1.2rem; padding: 0;">
                            {error_summary_lines}
                        </ul>
                        <p style="margin: 0; font-size: 0.86rem; color: #38bdf8;">
                            👉 <strong>Next Steps:</strong> Wait 10-15 seconds and click <em>Retry Question</em> below, or add an OpenAI / Anthropic key in the sidebar for backup routing!
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
