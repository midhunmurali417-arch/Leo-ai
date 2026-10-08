"""
=============================================================================
Module: google_search.py
Description: Real-time Google Search & Live News Grounding Engine for Leo AI Chatbot
Features:
  1. Official Google Custom Search JSON API integration (Programmable Search Engine)
  2. Real-Time Google News Live Search API (Zero-Key RSS feed with timestamps)
  3. Automatic intelligent query intent detection (news, current events, 2026 data)
  4. Robust fallback handling (if Google Custom Search quota or key expires,
     seamlessly falls back to Google News live search)
  5. Formatted markdown citations & prompt injection builder for LLM grounding
=============================================================================
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Common keywords that indicate real-time / current / news intent
TEMPORAL_KEYWORDS = {
    "latest", "news", "today", "yesterday", "current", "currently", "recent",
    "recently", "update", "updates", "breaking", "now", "live", "price",
    "score", "winner", "election", "weather", "released", "release date",
    "newest", "happened", "happening", "upcoming", "2025", "2026", "2027",
    "who is", "who won", "president", "ceo", "prime minister", "stock"
}


def clean_search_query(query: str) -> str:
    """Cleans conversational filler words from user prompts to optimize search relevance."""
    clean = query.strip()
    # Strip common chatbot question preambles
    patterns = [
        r"^(can you\s+)?(please\s+)?(tell me\s+)?(what is|what are|who is|who are)\s+the\s+latest\s+",
        r"^(can you\s+)?(please\s+)?(tell me\s+)?(about\s+)?the\s+latest\s+",
        r"^(can you\s+)?(please\s+)?(give me\s+)?(news\s+about|updates\s+on)\s+",
        r"^(what('s| is)\s+happening\s+with)\s+",
        r"^(search\s+(google\s+for|for)\s+)",
    ]
    for pat in patterns:
        clean = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()

    # Strip trailing punctuation
    clean = clean.rstrip("?").rstrip("!").rstrip(".").strip()
    return clean if clean else query.strip()


def should_search_google(query: str) -> bool:
    """
    Intelligently determines if a user query requires real-time Google search grounding.
    Returns True for questions about recent events, current facts, news, or temporal queries.
    """
    q_lower = query.lower()

    # Check for direct keywords
    words = re.findall(r"\b\w+\b", q_lower)
    if any(w in TEMPORAL_KEYWORDS for w in words):
        return True

    # Check for phrases
    phrases = [
        "what happened", "what is happening", "right now", "this week",
        "this month", "this year", "current date", "current price",
        "who is the current", "latest news", "breaking news", "new release"
    ]
    if any(p in q_lower for p in phrases):
        return True

    return False


def search_google_custom_search(
    query: str,
    api_key: str,
    cse_id: str,
    num_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Queries Google Custom Search JSON API (Programmable Search Engine).
    API Docs: https://developers.google.com/custom-search/v1/overview
    """
    if not api_key or not cse_id:
        return []

    encoded_q = urllib.parse.quote(query)
    num = max(1, min(10, num_results))
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={encoded_q}&num={num}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LeoAIChatbot/2.0 (Streamlit; Python 3.10+)",
            "Accept": "application/json"
        }
    )

    with urllib.request.urlopen(req, timeout=5.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    items = data.get("items", [])
    for item in items:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        display_link = item.get("displayLink", "")

        # Extract publication date if present in pagemap metatags
        published = ""
        pagemap = item.get("pagemap", {})
        metatags = pagemap.get("metatags", [])
        if metatags and isinstance(metatags, list) and len(metatags) > 0:
            first_meta = metatags[0]
            published = (
                first_meta.get("article:published_time")
                or first_meta.get("date")
                or first_meta.get("og:updated_time")
                or ""
            )
            if published and "T" in published:
                published = published.split("T")[0]

        results.append({
            "title": title,
            "snippet": snippet,
            "url": link,
            "source": display_link,
            "published": published,
            "type": "web"
        })

    return results


def search_google_news(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Queries real-time Google News RSS search feed.
    Provides up-to-the-minute verified news articles with headlines, publishers,
    and timestamps without requiring an API key.
    """
    clean_q = clean_search_query(query)
    encoded_q = urllib.parse.quote(clean_q)
    url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml"
        }
    )

    with urllib.request.urlopen(req, timeout=5.0) as resp:
        xml_content = resp.read().decode("utf-8")

    root = ET.fromstring(xml_content)
    items = root.findall(".//item")

    results = []
    limit = max(1, min(10, num_results))

    for item in items[:limit]:
        raw_title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or ""
        source = item.findtext("source") or ""

        # Parse cleaner title & source if formatted as 'Headline - Publisher'
        clean_title = raw_title
        if " - " in raw_title:
            parts = raw_title.rsplit(" - ", 1)
            clean_title = parts[0].strip()
            if not source:
                source = parts[1].strip()

        # Format publication date nicely
        formatted_date = pub_date
        try:
            # Typical format: 'Tue, 06 Oct 2026 13:19:01 GMT'
            dt = datetime.strptime(pub_date[:16], "%a, %d %b %Y")
            formatted_date = dt.strftime("%b %d, %Y")
        except Exception:
            formatted_date = pub_date[:16] if pub_date else ""

        # Clean description HTML if present
        description = item.findtext("description") or ""
        clean_snippet = re.sub(r"<[^>]+>", " ", description).strip()
        if not clean_snippet:
            clean_snippet = clean_title

        results.append({
            "title": clean_title,
            "snippet": clean_snippet,
            "url": link,
            "source": source or "Google News",
            "published": formatted_date,
            "type": "news"
        })

    return results


def execute_google_search(
    query: str,
    api_key: str = "",
    cse_id: str = "",
    num_results: int = 5,
    mode: str = "auto"
) -> Dict[str, Any]:
    """
    Unified Google Search Execution Engine.
    Intelligently routes across Google Custom Search API and Google News Live Search:
      - Uses Custom Search JSON API if configured
      - Uses Google News Live Search for news/breaking/temporal questions or fallback
      - Merges and deduplicates results into a unified context payload
    """
    clean_q = clean_search_query(query)
    has_cse_creds = bool(api_key and cse_id and api_key != "your_google_search_api_key_here")

    results = []
    source_title = "Google News Live Feed"
    errors = []

    # 1. Attempt Google Custom Search if credentials exist
    if has_cse_creds and mode != "news":
        try:
            cse_res = search_google_custom_search(clean_q, api_key, cse_id, num_results=num_results)
            if cse_res:
                results.extend(cse_res)
                source_title = "Google Custom Search API"
        except Exception as e:
            errors.append(f"Custom Search: {str(e)}")

    # 2. If results are empty or mode is news or temporal query, query Google News
    is_temporal = should_search_google(query)
    if not results or mode == "news" or (is_temporal and len(results) < num_results):
        try:
            news_res = search_google_news(clean_q, num_results=num_results)
            if news_res:
                if results:
                    # Blend Custom Search + Google News
                    existing_urls = {r["url"] for r in results}
                    for nr in news_res:
                        if nr["url"] not in existing_urls:
                            results.append(nr)
                    source_title = "Google Custom Search + Google News"
                else:
                    results = news_res
                    source_title = "Google News Live Search"
        except Exception as e:
            errors.append(f"Google News: {str(e)}")

    # Trim to requested number
    final_results = results[:num_results]

    # Build structured text for LLM injection
    prompt_lines = []
    if final_results:
        prompt_lines.append(f"--- REAL-TIME GOOGLE SEARCH & NEWS RESULTS (Search Query: \"{clean_q}\") ---")
        prompt_lines.append(f"Engine: {source_title} | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for idx, item in enumerate(final_results, 1):
            date_str = f" ({item['published']})" if item.get("published") else ""
            source_str = f" [{item.get('source')}]" if item.get("source") else ""
            prompt_lines.append(f"{idx}. {item.get('title')}{source_str}{date_str}")
            if item.get("snippet"):
                prompt_lines.append(f"   Snippet: {item.get('snippet')}")
            if item.get("url"):
                prompt_lines.append(f"   URL: {item.get('url')}")
            prompt_lines.append("")
        prompt_lines.append("--- END GOOGLE SEARCH RESULTS ---")

    prompt_text = "\n".join(prompt_lines)

    return {
        "query": clean_q,
        "source_title": source_title,
        "results": final_results,
        "prompt_text": prompt_text,
        "errors": errors if errors else None
    }


def test_google_search_connection(api_key: str = "", cse_id: str = "") -> Tuple[int, str]:
    """
    Tests Google Search connectivity and measures latency in milliseconds.
    Tests Custom Search API if credentials provided, otherwise tests Google News RSS feed.
    """
    t0 = time.time()
    if api_key and cse_id and api_key != "your_google_search_api_key_here":
        res = search_google_custom_search("Google AI latest", api_key, cse_id, num_results=1)
        latency = round((time.time() - t0) * 1000)
        return latency, "Official Google Custom Search JSON API"
    else:
        res = search_google_news("Google AI latest", num_results=1)
        latency = round((time.time() - t0) * 1000)
        return latency, "Google News Live Real-Time Feed"
