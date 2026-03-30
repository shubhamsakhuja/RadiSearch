# Changelog

All notable changes to RadiSearch are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — 2026-03-17

### Added
- **Microsoft Azure AD SSO** — login with existing hospital Microsoft accounts via OAuth2/OIDC
- **Dev mode** — `DEV_MODE=true` in `.env` bypasses Azure AD for local testing without credentials
- **User display** — logged-in user's name and Logout button shown in topbar on all pages
- **Modular project structure** — monolithic `app.py` split into 12 focused modules across `routes/` and `templates/`
- **`/ping` endpoint** — browser calls this on every user interaction to reset the inactivity timer
- **Inactivity timer fix** — timer now resets on any button click (search, reset, filter, export, tab switch, history pill, expand/collapse), not just on search queries
- **`requirements.txt`** — single-command dependency installation
- **`.env.example`** — safe credentials template for new developers
- **`CHANGELOG.md`**, **`CONTRIBUTING.md`**, **`README.md`** — full project documentation with architecture diagrams

### Changed
- All HTML templates moved from Python string constants to proper Jinja2 `.html` files in `templates/`
- `render_template_string()` replaced with `render_template()` throughout
- LLM mode page theme updated from dark to light — now consistent with Custom Search page
- `last_activity` timer ownership moved to `routes/api.py` with a clean `reset_activity()` API
- Flask `app` object and Blueprint registration centralised in `app.py`
- `search_history` list moved to `routes/search_routes.py` (closer to where it is written)
- `chat_sessions` dict moved to `routes/llm_routes.py`

### Fixed
- Inactivity timer no longer resets when navigating between pages (only real actions reset it)
- Corrupt or stale embedding cache now caught and rebuilt gracefully instead of crashing
- PostgreSQL connection now always closed in `finally` block — prevents connection pool exhaustion
- `/search_mode` route decorator was previously missing — restored

---

## [1.0.0] — 2026-01-15

### Added
- Initial release as a single-file Flask application (`app.py`)
- **Custom Search mode** — semantic search using FAISS + sentence-transformers
- **LLM mode** — natural language queries via local Ollama AI model
- **Radiologist Report Metrics panel** — filter by radiologist, modality, clinic, date range
- **Dashboard tab** — report stats, modality bar chart, top radiologists, 30-day trend sparkline
- **Search history pills** — last 20 queries shown as clickable shortcuts
- **Semantic search** with `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **FAISS IndexFlatIP** with cosine similarity (L2-normalised vectors)
- **Keyword fallback** — regex scan when semantic search returns fewer than 5 results
- **Daily embedding cache** — FAISS index rebuilt once per day, cached between restarts
- **Noon background refresh** — embeddings silently rebuilt at 12:00 each day
- **Batch encoding** — reports encoded in batches of 64 for memory efficiency
- **Similarity threshold** — 40% minimum cosine similarity to filter irrelevant results
- **Two-path search** — fast path (global pre-built index) and filtered path (temporary sub-index)
- **LLM intent parsing** — Ollama extracts structured parameters from natural language
- **Server-Sent Events streaming** — AI response appears word-by-word in real time
- **Conversation memory** — last 20 turns stored per browser session
- **Export CSV** — client-side CSV export of any result set
- **Expand/Collapse All** — bulk toggle for report detail sections
- **Inactivity auto-shutdown** — server closes after 5 minutes of no activity
- **Loading overlay** with real-time progress bar during embedding build
- **`/closed` page** — clean shutdown message with restart instructions
