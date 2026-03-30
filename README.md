# RadiSearch 🔍

> **Plain English:** RadiSearch lets hospital staff search through thousands of radiology reports by typing natural questions — like asking a smart assistant. It understands medical language, typos, casual phrasing, and follow-up questions. Runs entirely inside your hospital network by default.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org) [![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com) [![ChromaDB](https://img.shields.io/badge/Vector_Store-ChromaDB-orange.svg)](https://www.trychroma.com) [![AWS](https://img.shields.io/badge/Cloud-AWS_Bedrock-yellow.svg)](https://aws.amazon.com/bedrock/) [![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## What Does It Do?

Imagine 100,000 radiology reports in a hospital database. A radiologist wants all CT chest reports mentioning pleural effusion from last month. Normally that means calling IT for a SQL query.

RadiSearch lets them type it in plain English and get the answer in seconds.

```
You type:  "Show me CT chest reports mentioning pleural effusion last month for Dr Smith"
                                    ↓
            AI understands your intent (even typos and casual language)
                                    ↓
            Searches 100,000 reports instantly
                                    ↓
            Shows matching reports + AI summary
```

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | Finds reports by *meaning*. "Heart attack" also finds "MI", "myocardial infarction", "cardiac event" |
| 🤖 **AI Chat (LLM Mode)** | Ask anything in plain English — search, summarise, rank, compare, or analyse |
| 📊 **Analytics** | "Rank radiologists by report count", "monthly trend by modality", "busiest clinic per month" |
| 🧠 **Fuzzy Understanding** | Handles typos, abbreviations, casual terms: "catscam" → CT, "xray" → CR, "smyth" → SMITH |
| 🔄 **Follow-up Queries** | Understands conversation context: "same but last month", "now filter by CT", "top 10 only" |
| 🕐 **Time Intelligence** | "recently", "last financial year", "morning shift", "Q3", "Australian summer" |
| 👤 **Radiologist Filter** | Filter by doctor, modality, clinic, date range, exam code |
| 📋 **Summarise** | Clinical summaries across any set of reports |
| 🏆 **Rank** | Rank reports by clinical relevance to a topic |
| 🔀 **Compare** | "CT vs MRI volumes", "this month vs last month", "Dr Smith vs Dr Jones" |
| 💾 **Export CSV** | Download any result set as a spreadsheet |
| 🔐 **Hospital Login** | Microsoft Azure AD SSO — existing work account, no new password |
| ☁️ **AWS Bedrock** | Optional cloud AI (Claude 3 Haiku) instead of local Ollama |
| 🔒 **Privacy First** | Runs inside your network by default — no data leaves |

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR HOSPITAL NETWORK                        │
│                                                                 │
│  Browser          Flask Server              AI Layer            │
│  ┌──────┐        ┌──────────────┐     ┌──────────────────┐    │
│  │ You  │──────▶ │  RadiSearch  │────▶│ Ollama (local)   │    │
│  │      │        │  app.py      │     │   OR             │    │
│  └──────┘        └──────┬───────┘     │ AWS Bedrock      │    │
│                         │             └──────────────────┘    │
│                    ┌────┴──────────────────┐                   │
│              ┌─────▼──────┐    ┌───────────▼──┐               │
│              │ PostgreSQL │    │  ChromaDB    │               │
│              │ (reports)  │    │ (search index│               │
│              └────────────┘    └──────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### LLM Mode Pipeline

Every message goes through a 6-step pipeline:

```
1. INTENT PARSING    — AI extracts: task, filters, dates, clinical terms
2. FUZZY RESOLUTION  — Normalise: XR→CR, "smyth"→SMITH, "emergancy"→EMERGENCY DEPT
3. DATABASE SEARCH   — ChromaDB semantic search + in-memory filtering
4. ANALYTICS ENGINE  — For counts/rankings: generate & run pandas code dynamically
5. CONTEXT BUILDING  — Format records for the AI (full text or metadata by task)
6. AI RESPONSE       — Stream answer word-by-word via Server-Sent Events
```

### Task Types

| Task | Triggers | Example |
|---|---|---|
| **search** | show, find, get, give me | "Show me fracture reports this week" |
| **summarise** | summarise, overview, what findings | "Summarise CT chest findings from Emergency" |
| **answer** | what, which, how, tell me | "What conditions appear most in MRI reports?" |
| **analytics** | rank, count, busiest, trend, breakdown | "Rank radiologists by XR report count" |
| **compare** | compare, vs, versus | "Compare CT and MRI volumes by clinic" |
| **rank** | rank reports, most relevant | "Rank reports most relevant to pleural effusion" |

### Analytics Engine

For analytics queries, RadiSearch dynamically generates and executes pandas code:

```
User: "Rank all radiologists by XR report count per month"
         ↓
Intent parser → task=analytics, modality=CR (XR resolved to CR)
         ↓
run_search() → 20,025 CR records from meta_data (instant)
         ↓
execute_analytics() → LLM writes:
  result = df.groupby(['radiologist', df['report_date'].dt.strftime('%Y-%m')])
             .size().reset_index(name='count')
             .sort_values('count', ascending=False)
         ↓
exec() runs it safely → compact markdown table (~500 tokens)
         ↓
Final AI response streams the table with insights
```

This handles any aggregation — no hardcoded groupings.

### Fuzzy Resolution Layer

`fuzzy.py` runs between intent parsing and search, normalising all values:

| User says | Resolved to |
|---|---|
| "xray" / "XR" / "x-ray" | CR (Computed Radiography) |
| "catscam" / "cat scan" | CT |
| "mammo" | MG |
| "smyth" (typo) | SMITH, JAMES (fuzzy match) |
| "emergancy" (typo) | EMERGENCY DEPT |
| "heart attack" | myocardial infarction MI cardiac |
| "broken bone" | fracture break |
| "recently" | last 7 days |
| "financial year" | Jul 1 – Jun 30 (Australian FY) |
| "morning shift" | visit_datetime 06:00–11:59 |
| "busiest doctor" | analytics: group by radiologist, sort desc |

---

## Semantic Search

Normal search finds exact words. Semantic search finds meaning.

```
You search: "heart attack"

Traditional:    Only finds "heart attack" ❌
RadiSearch:     Also finds "myocardial infarction", "MI",
                "cardiac arrest", "acute coronary event" ✅
```

Every report is converted to a 384-dimensional vector (embedding) representing its meaning. Searches find reports with the closest meaning, not just matching words. Stored permanently in ChromaDB — loads in 2–5 seconds after first build.

---

## AWS Integration

Two optional AWS services — both opt-in, app works fully without them.

### Bedrock LLM (replaces Ollama)

| | Local Ollama | AWS Bedrock |
|---|---|---|
| Cost | Free | ~$0.002/query |
| Speed | 30–90 seconds | 3–10 seconds |
| Setup | Install Ollama | AWS account + API keys |
| Works offline | ✅ | ❌ |
| Data leaves network | ❌ | ✅ (query text only) |

Available models: Claude 3 Haiku (~$0.002), Claude 3.5 Haiku (~$0.004), Claude 3 Sonnet (~$0.015)

**One-time setup:** AWS Console → Bedrock → Model access → enable Claude 3 Haiku

### S3 Data Source (optional)

Store `embedding_meta.pkl` in S3 instead of local disk — useful for EC2 deployments.

### Estimated Cost (personal/portfolio use)

```
First 12 months (AWS free tier):
  Bedrock AI:   ~$1.20/month  (20 queries/day × $0.002)
  EC2:          $0.00         (free tier)
  S3:           $0.00         (free tier)

After 12 months:
  Bedrock AI:   ~$1.20/month
  EC2 t3.micro: ~$8.47/month
  Total:        ~$9.70/month
```

---

## Project Structure

```
radisearch/
├── app.py              ← Entry point — run this
├── config.py           ← All settings — edit .env not this
├── auth.py             ← Microsoft Azure AD SSO
├── database.py         ← PostgreSQL connection
├── embeddings.py       ← ChromaDB vector index management
├── search.py           ← Search logic (semantic + filter + fallback)
├── llm.py              ← LLM integration (Ollama + Bedrock), analytics engine
├── fuzzy.py            ← Fuzzy resolution layer (typos, aliases, dates)
│
├── routes/
│   ├── __init__.py
│   ├── api.py          ← /heartbeat, /ping, /stats, /progress
│   ├── pages.py        ← HTML page routes
│   ├── search_routes.py← /search, /filter_reports
│   └── llm_routes.py   ← /llm_query, /clear_session
│
├── templates/
│   ├── landing.html    ← Mode selection
│   ├── search_mode.html← Custom search + dashboard
│   └── llm_mode.html   ← AI chat interface
│
├── .env                ← Credentials (NEVER commit)
├── requirements.txt    ← Python dependencies
├── embedding_meta.pkl  ← Report data (not in git)
└── chroma_db/          ← Search index on disk (not in git)
```

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/yourusername/radisearch.git
cd radisearch

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Mac/Linux

# 3. Install packages
pip install -r requirements.txt

# 4. Create .env with minimum settings
echo "DEV_MODE=true" > .env
echo "USE_PKL_DATA=true" >> .env
echo "FLASK_SECRET_KEY=localtest123" >> .env

# 5. Generate dummy data (100,000 sample reports)
python generate_large.py

# 6. Run
python app.py
# Browser opens at http://127.0.0.1:5000
```

First launch builds the ChromaDB index (~20–30 min for 100k records). Every subsequent launch loads from disk in 2–5 seconds.

---

## Configuration Reference

All settings in `.env`:

```env
# ── Data source ───────────────────────────────────────────────────
USE_PKL_DATA=true          # true=pkl file, false=PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_DATABASE=hospital_db
POSTGRES_USERNAME=admin
POSTGRES_PASSWORD=secret

# ── Authentication ────────────────────────────────────────────────
DEV_MODE=true              # true=skip login (testing only!)
AZURE_CLIENT_ID=xxxx
AZURE_CLIENT_SECRET=xxxx
AZURE_TENANT_ID=xxxx
AZURE_ALLOWED_DOMAIN=hospital.com

# ── Flask ─────────────────────────────────────────────────────────
FLASK_SECRET_KEY=generate-with-secrets.token_hex(32)

# ── AWS (only needed when USE_BEDROCK=true or USE_S3=true) ────────
AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=your-secret-key

# ── Bedrock LLM ───────────────────────────────────────────────────
USE_BEDROCK=false          # false=Ollama (free), true=AWS Bedrock
BEDROCK_LLM_MODEL=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_MAX_TOKENS=4096

# ── Bedrock Embeddings (leave false unless committing long-term) ──
USE_BEDROCK_EMBED=false    # Changing this requires deleting chroma_db/
BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0

# ── S3 (optional, for EC2 deployments) ───────────────────────────
USE_S3=false
AWS_S3_BUCKET=
AWS_S3_PKL_KEY=embedding_meta.pkl
```

---

## Security

- **Microsoft SSO** — only `@yourhospital.com` accounts admitted
- **HttpOnly + SameSite cookies** — XSS/CSRF protection
- **Inactivity shutdown** — server closes after 15 minutes idle, clearing RAM
- **No credentials in code** — all secrets in `.env`, never committed to git
- **AWS Bedrock** — does not train on your data; query text only, not stored

---

## Tech Stack

| Component | Purpose |
|---|---|
| Flask | Web framework, routing, SSE streaming |
| ChromaDB | Persistent vector database for semantic search |
| sentence-transformers | Local embedding model (all-MiniLM-L6-v2, 384-dim) |
| pandas | In-memory analytics, dynamic code execution |
| Ollama | Local AI model runner |
| AWS Bedrock | Cloud LLM (Claude 3 Haiku) |
| MSAL | Microsoft Azure AD OAuth2 |
| psycopg2 | PostgreSQL driver |
| boto3 | AWS SDK |
| python-dotenv | .env loading |

---

*Built by Shubham Sakhuja — Data Analytics Leader, healthcare AI and AWS cloud architecture.*
