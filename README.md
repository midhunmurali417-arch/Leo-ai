# Leo 🦁 - AI Assistant Chatbot (MCA LLM Project)

A complete, production-ready conversational AI assistant built with **Python**, **Streamlit**, and **Google Gemini 2.0 Flash** via the `google-generativeai` SDK, featuring an **animated Cartoon AI Lion Mascot** on screen!

---

## 📁 Project Folder Structure

```text
LLM_PROJECT/
│
├── app.py                # Main Streamlit application source code
├── google_search.py      # Real-time Google Search & Live News Grounding Engine
├── requirements.txt      # Python dependencies list
├── assets/               # Mascot and avatar assets (cartoon_lion.jpg)
├── .env.example          # Environment variables template
├── .env                  # Your private API key file (create this from .env.example)
└── README.md             # Project documentation & viva guide
```

---

## 🚀 Key Features

1. **Cartoon AI Lion Character on Screen**:
   - High-fidelity 3D cartoon AI lion mascot with sleek glowing cyan headphones and friendly smile.
   - Interactive comic speech bubble next to Leo with dynamic expressions.
   - Floating animated cartoon companion on the screen with hover effects.
   - Custom cartoon lion avatars on every chat message.
2. **Real-Time Google Search & Live News Grounding Engine**:
   - **Official Google Custom Search JSON API**: Connects via Google Cloud API Key and Search Engine ID (CX) to query live web data.
   - **Google News Real-Time Search API**: Out-of-the-box live news feed with headlines, sources (BBC, Reuters, etc.), timestamps, and links—works instantly with **zero API keys required**!
   - **Intelligent Query Intent Detection**: Automatically detects questions about latest news, current facts, 2026 events, prices, and scores.
   - **Interactive Citations & Source Cards**: Expandable source verification cards showing real article links and publication dates.
3. **Multi-Provider AI Architecture**:
   - **Google Gemini**: Gemini 3 Flash, Gemini Flash Latest, Gemini Flash-Lite, Gemini 3.5 Flash.
   - **OpenAI ChatGPT**: GPT-4o Mini (fast & smart), GPT-4o (flagship).
   - **Anthropic Claude**: Claude 3.5 Haiku (lightning fast), Claude 3.5 Sonnet (state-of-the-art).
4. **Smart Multi-Provider Rate-Limit Fallback Shield**:
   - When Google Gemini hits free-tier rate limits (15 RPM or daily quota), Leo automatically and seamlessly routes the question to **ChatGPT (GPT-4o Mini)** or **Claude (Claude 3.5 Haiku)**!
   - 1-Click Retry button so you never have to re-type a question if cooldown occurs.
   - Multi-key manager in sidebar with live connection test and latency indicator for all providers.
5. **Ultra-Fast Zero-Delay Streaming**:
   - Yields token-by-token streaming across all providers via `st.write_stream`.
   - Sliding memory window to keep latency minimal and prevent token bloat.
6. **Multi-Turn Conversational Memory**: Configurable memory window in the sidebar.
7. **Interactive Controls**:
   - Google Search & Live Grounding toggle with Smart Auto / Always / News Only modes.
   - Primary model selector across Gemini, OpenAI, and Anthropic.
   - Creativity slider (temperature) and Accuracy Mode presets.
   - Memory window slider.
   - Floating cartoon lion toggle.
   - Chat export (.txt) and instant reset.

---

## 🛠️ Prerequisites

- **Python 3.10** or higher
- At least one AI API key:
  - **Google Gemini**: Free at [Google AI Studio](https://aistudio.google.com/app/apikey)
  - **OpenAI (ChatGPT)**: Optional fallback at [OpenAI Platform](https://platform.openai.com/api-keys)
  - **Anthropic (Claude)**: Optional fallback at [Anthropic Console](https://console.anthropic.com/settings/keys)
- *(Optional)* For Google Custom Search API:
  - **Google Custom Search API Key**: [Google Cloud Console](https://console.cloud.google.com/apis/library/customsearch.googleapis.com)
  - **Search Engine ID (CX)**: [Programmable Search Engine](https://programmablesearchengine.google.com/)
  - *Note: Live Google News search operates automatically even without these search keys!*

---

## ⚙️ Step-by-Step Setup & Installation

### 1. Open Terminal / PowerShell in the Project Directory
Navigate to the project folder:
```powershell
cd c:\Users\midhu\OneDrive\Desktop\LLM_PROJECT
```

### 2. Create a Virtual Environment (Recommended)
Creating an isolated environment keeps dependencies clean:

**On Windows (PowerShell):**
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Keys
1. Copy `.env.example` to a new file named `.env`:
   - On Windows PowerShell:
     ```powershell
     Copy-Item .env.example .env
     ```
2. Open `.env` and paste your API key(s):
   ```env
   # Google Gemini API Key (Free tier)
   GEMINI_API_KEY=your_gemini_api_key_here

   # OpenAI API Key (ChatGPT Fallback - Optional)
   OPENAI_API_KEY=your_openai_api_key_here

   # Anthropic Claude API Key (Claude Fallback - Optional)
   ANTHROPIC_API_KEY=your_anthropic_api_key_here

   # Google Custom Search API Key & Search Engine ID (Optional)
   GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
   GOOGLE_CSE_ID=your_google_cse_id_here
   ```
*(You can also enter or update keys directly in the app's sidebar under "Multi-Provider API Keys" and click "Save Keys" to persist them into `.env` automatically).*

---

## ▶️ Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

Streamlit will automatically open your default web browser at:
```text
Local URL: http://localhost:8501
```

---

## 🎓 Viva & Technical Explanation Guide

Use this section to prepare for viva questions and project presentations:

### 1. How does Real-Time Google Search Grounding work?
- **The Problem with LLMs (Knowledge Cutoff & Hallucination)**: LLMs have static knowledge cutoffs and cannot know recent news or current events without external grounding.
- **Leo's Dual-Layer Google Search Pipeline (`google_search.py`)**:
  1. **Intent Detection**: `should_search_google(query)` detects keywords like *"latest"*, *"today"*, *"news"*, *"breaking"*, *"2026"*, *"current"*, or specific names/scores.
  2. **Official Google Custom Search JSON API**: If credentials (`GOOGLE_SEARCH_API_KEY` & `GOOGLE_CSE_ID`) are configured, Leo queries `https://www.googleapis.com/customsearch/v1` for targeted web pages and snippets.
  3. **Zero-Key Real-Time Google News Feed**: If credentials are absent or if the query requests news, Leo queries `news.google.com/rss/search` to retrieve up-to-the-minute headlines, timestamps, publishers, and links.
  4. **Grounding Injection**: The retrieved factual context is injected into the LLM system prompt as verified evidence before the LLM generates its response.
  5. **Transparent Citations**: Leo renders an expandable citation box with clickable source links and publication dates above the assistant answer.

### 2. How does Multi-Provider Rate Limit Fallback work?
- Free tier AI APIs (such as Google Gemini) impose quotas (e.g. 5–15 requests/minute).
- Leo implements an **Automatic Failover Shield**:
  1. The user selects a primary model (e.g., `Gemini 3 Flash`).
  2. If the primary provider returns HTTP 429 (`ResourceExhausted` or `RateLimitError`), the app catches this exception before the UI crashes.
  3. Leo automatically cascades to backup Gemini models, or to **ChatGPT (`gpt-4o-mini`)** or **Claude (`claude-3-5-haiku`)**.
  4. The user receives a seamless response with an informative badge notifying them of the failover.

### 3. How does Conversation Memory work in Streamlit?
- **Stateless Nature of Streamlit**: Whenever a user interacts with a Streamlit widget, Streamlit reruns the Python script from top to bottom.
- **The Solution (`st.session_state`)**: `st.session_state` is a dictionary-like session store that persists across script reruns for the same browser tab.
- **Implementation**:
  ```python
  if "messages" not in st.session_state:
      st.session_state.messages = []
  ```
  Every message (and its optional search grounding metadata) is stored and formatted before sending to the active model provider.

### 4. How is Unified Streaming implemented across multiple providers?
- **Gemini**: Consumes the generator from `chat.send_message(..., stream=True)` yielding `chunk.text`.
- **OpenAI**: Consumes `client.chat.completions.create(..., stream=True)` yielding `chunk.choices[0].delta.content`.
- **Anthropic**: Consumes `client.messages.stream(...)` yielding tokens from `stream.text_stream`.
- All three feed into Streamlit's `st.write_stream()`, providing a smooth, uniform streaming experience regardless of which LLM answered.

### 5. What does the Temperature parameter do?
- **Temperature** controls randomness during token sampling (softmax):
  - **Low Temperature (0.0 - 0.3 / High Accuracy Mode)**: Highly deterministic, factual, grounded answers.
  - **Medium Temperature (0.5 - 0.7)**: Optimal balance of creativity and coherence.
  - **High Temperature (0.8 - 1.0)**: Maximum novelty and diverse phrasing.
