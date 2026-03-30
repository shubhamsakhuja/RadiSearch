# RadiSearch — Complete Setup & Run Guide

---

## Project Structure

```
radisearch/
├── app.py              ← the only file you run
├── config.py           ← all settings read from .env
├── auth.py             ← Azure AD login
├── database.py         ← PostgreSQL connection
├── embeddings.py       ← ChromaDB vector index
├── search.py           ← search logic (semantic + filter + fallback)
├── llm.py              ← LLM integration + analytics engine
├── fuzzy.py            ← fuzzy resolution (typos, aliases, dates) ← NEW
├── routes/
│   ├── __init__.py
│   ├── api.py
│   ├── pages.py
│   ├── search_routes.py
│   └── llm_routes.py
├── templates/
│   ├── landing.html
│   ├── search_mode.html
│   └── llm_mode.html
├── .env                ← your credentials (never commit)
├── requirements.txt
├── embedding_meta.pkl  ← your report data
└── chroma_db/          ← search index (auto-created)
```

---

## Complete .env Variable Reference

### Minimum to run locally (3 lines)

```env
DEV_MODE=true
USE_PKL_DATA=true
FLASK_SECRET_KEY=any-string-for-local-testing
```

### Full variable list

```env
# ════════════════════════════════════════════════════════
# POSTGRESQL — only when USE_PKL_DATA=false
# ════════════════════════════════════════════════════════
POSTGRES_HOST=localhost
POSTGRES_DATABASE=hospital_db
POSTGRES_USERNAME=admin
POSTGRES_PASSWORD=your_password

# ════════════════════════════════════════════════════════
# DATA SOURCE
# ════════════════════════════════════════════════════════
# true  → load from embedding_meta.pkl (testing)
# false → connect to PostgreSQL (production)
USE_PKL_DATA=true

# ════════════════════════════════════════════════════════
# FLASK
# ════════════════════════════════════════════════════════
# Generate: python -c "import secrets; print(secrets.token_hex(32))"
FLASK_SECRET_KEY=your-long-random-string

# ════════════════════════════════════════════════════════
# DEV MODE
# ════════════════════════════════════════════════════════
# true  → skip Azure AD, auto-login as Dev User (local testing ONLY)
# false → require Microsoft login (production)
DEV_MODE=true

# ════════════════════════════════════════════════════════
# AZURE AD SSO — only when DEV_MODE=false
# ════════════════════════════════════════════════════════
AZURE_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_CLIENT_SECRET=your-secret-from-azure
AZURE_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_ALLOWED_DOMAIN=hospital.com

# ════════════════════════════════════════════════════════
# AWS CREDENTIALS — only when USE_BEDROCK=true or USE_S3=true
# ════════════════════════════════════════════════════════
# Get from: AWS Console → Security credentials → Access keys → Create
AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=your-secret-key

# ════════════════════════════════════════════════════════
# BEDROCK LLM — replaces Ollama when USE_BEDROCK=true
# ════════════════════════════════════════════════════════
# false → Ollama (free, local)
# true  → AWS Bedrock Claude (~$0.002/query)
# One-time setup: AWS Console → Bedrock → Model access → enable Claude 3 Haiku
USE_BEDROCK=false
BEDROCK_LLM_MODEL=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_MAX_TOKENS=4096

# ════════════════════════════════════════════════════════
# BEDROCK EMBEDDINGS — replaces sentence-transformers
# ════════════════════════════════════════════════════════
# WARNING: switching requires deleting chroma_db/ and full rebuild
# false → local all-MiniLM-L6-v2 (free, default)
# true  → Amazon Titan V2 (~$0.02/1M tokens)
USE_BEDROCK_EMBED=false
BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0

# ════════════════════════════════════════════════════════
# S3 — optional, for EC2 deployments only
# ════════════════════════════════════════════════════════
USE_S3=false
AWS_S3_BUCKET=
AWS_S3_PKL_KEY=embedding_meta.pkl
```

---

## Scenario Runbooks

### Scenario 1 — Local testing with dummy data (free, no accounts)

**Steps:**

```bash
# 1. Virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env
DEV_MODE=true
USE_PKL_DATA=true
FLASK_SECRET_KEY=localtest123

# 4. Generate dummy data
python generate_large.py

# 5. Run
python app.py
```

Expected terminal output:
```
  DEV_MODE         : True
  USE_PKL_DATA     : True
  USE_BEDROCK      : False
  USE_BEDROCK_EMBED: False
  USE_S3           : False
  embedding_meta.pkl  : 100,000 records
  chroma_db/          : not found (index will be built — first run)

[config] Bedrock LLM  : disabled — using local Ollama
```

First run builds ChromaDB index (~20–30 min). Every subsequent run loads in 2–5 seconds.

---

### Scenario 2 — Local testing with AWS Bedrock LLM

**One-time AWS setup (do this first):**

1. **Get credentials:** AWS Console → top-right menu → Security credentials → Access keys → Create access key → Local code
2. **Enable model:** AWS Console → Bedrock → Model access → find Claude 3 Haiku → Request access (usually instant)
3. **Fill .env:**

```env
DEV_MODE=true
USE_PKL_DATA=true
FLASK_SECRET_KEY=localtest123

AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=your-secret-key

USE_BEDROCK=true
USE_BEDROCK_EMBED=false   ← keep false, local embeddings are free
USE_S3=false
```

Expected terminal output:
```
[config] Bedrock LLM  : ENABLED  — anthropic.claude-3-haiku-20240307-v1:0  (ap-southeast-2)
[config] Bedrock Embed: disabled — using local sentence-transformers
[config] S3 data      : disabled — using local embedding_meta.pkl
```

**Cost check:** AWS Console → Billing → Cost Explorer → filter by Bedrock. 50 test queries ≈ $0.10.

---

### Scenario 3 — Full production (PostgreSQL + Azure AD + Bedrock)

```env
# Database
POSTGRES_HOST=your-db-server
POSTGRES_DATABASE=hospital_db
POSTGRES_USERNAME=admin
POSTGRES_PASSWORD=your_password
USE_PKL_DATA=false

# Auth
DEV_MODE=false
AZURE_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_CLIENT_SECRET=your-secret
AZURE_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_ALLOWED_DOMAIN=hospital.com

# Flask
FLASK_SECRET_KEY=generate-with-secrets.token_hex(32)

# AWS
AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=your-secret-key
USE_BEDROCK=true
USE_BEDROCK_EMBED=false
USE_S3=false
```

---

## Running the App

Always exactly one command:

```bash
python app.py
```

No flags or arguments. Everything is controlled through `.env`.

The app will:
1. Print startup summary showing active modes
2. Open browser at http://127.0.0.1:5000
3. Show loading bar while ChromaDB index builds or loads
4. Activate 15-minute inactivity timer once ready

To stop: press `Ctrl+C` or click Exit App in the browser.

---

## What the LLM Can Do

### Search
```
"Show me all CT reports"
"Find fracture reports this week"
"Get reports from Dr Smith in Emergency Dept"
"Give me XR chest reports from last month"
"Show reports mentioning DVT or PE"
```

### Summarise
```
"Summarise cancer findings in CT reports"
"What are common findings in chest X-rays?"
"Overview of MRI brain reports this year"
"Describe patterns in Emergency Dept reports"
```

### Answer
```
"What conditions appear most in MRI reports?"
"How many reports mention pleural effusion?"
"What findings are common in Dr Johnson's reports?"
"Which modality has the most abnormal findings?"
```

### Analytics
```
"Rank all radiologists by report count"
"How many XR reports per month?"
"Which clinic has the most CT scans?"
"Busiest radiologist last year"
"Report volume by modality this year"
"Top 5 exam codes in Emergency Dept"
"Monthly trend for MRI reports"
"Morning vs afternoon report volumes"
"Radiologist performance by clinic"
"Reports per day of week"
```

### Compare
```
"Compare CT and MRI volumes by clinic"
"This month vs last month"
"Dr Smith vs Dr Jones report counts"
"Compare Emergency Dept to Outpatient this year"
```

### Rank
```
"Rank reports most relevant to pleural effusion"
"Prioritise fracture reports by clinical significance"
"Top 5 most relevant cancer reports"
```

### Fuzzy / Casual Language
```
"catscam reports" → CT
"xray" / "XR" / "x-ray" → CR (Computed Radiography)
"smyths reports" → SMITH (fuzzy match)
"emergancy dept" → EMERGENCY DEPT
"heart attack" → myocardial infarction / MI / cardiac
"blood clot" → DVT / PE / thrombus
"recently" → last 7 days
"financial year" → Jul 1 – Jun 30 (Australian FY)
"morning shift" → visit_datetime 06:00–11:59
"busiest doctor" → analytics by radiologist count
```

### Follow-up Queries
```
(after CT search) "now show MRI instead"
(after results) "same but last month"
(after analytics) "top 10 only"
(after search) "now summarise them"
(after results) "filter to Emergency Dept only"
(after any query) "who was the busiest?"
```

---

## Modality Codes Reference

| Code | Meaning | User can also type |
|---|---|---|
| CT | Computed Tomography | cat scan, catscam, CT scan |
| MRI | Magnetic Resonance Imaging | MRI scan, magnetic, magnet |
| CR | Computed Radiography (X-Ray) | XR, xray, x-ray, plain film |
| DX | Digital X-Ray | digital xray |
| US | Ultrasound | ultrasound, sonogram, echo |
| NM | Nuclear Medicine | nuclear, nuc med, isotope |
| MG | Mammography | mammogram, mammo |
| XA | Angiography | angio, angiogram |
| RF | Fluoroscopy | fluoro, barium |
| PT | PET Scan | PET, positron |

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'USE_BEDROCK'` | Old config.py on disk | Replace config.py with latest version from outputs |
| `ModuleNotFoundError` | Package not installed | `pip install -r requirements.txt` |
| `FileNotFoundError: embedding_meta.pkl` | Missing data file | Run `python generate_large.py` |
| `AccessDeniedException` from AWS | Model not enabled | AWS Console → Bedrock → Model access → enable Claude 3 Haiku |
| `ResourceNotFoundException: use case details` | First-time AWS account | AWS Console → Bedrock → Model catalog → Claude 3 Haiku → fill use case form |
| `NoCredentialsError` | AWS keys missing | Check `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in .env |
| `ValidationException: prompt is too long` | Token limit exceeded | Already fixed — analytics uses pre-computed summaries |
| LLM says "field name mismatch" | Analytics re-filtering already-filtered data | Replace llm.py and llm_routes.py with latest versions |
| App shuts down after 15 min | Inactivity timer | Normal — restart with `python app.py` |
| Loading bar stuck | pkl file not found | Check `embedding_meta.pkl` exists in project root |
| "No reports found" on analytics | ChromaDB get() with no filters | Replace search.py — now reads from meta_data directly |
| SyntaxError in llm.py | String literal with real newline | Replace llm.py with latest version |

---

## Recommended Test Progression

```
Step 1  Verify basic run
        DEV_MODE=true, USE_PKL_DATA=true, USE_BEDROCK=false
        → python app.py → Custom Search → search "cancer" → expect results

Step 2  Verify LLM with Ollama
        Same as Step 1 + Ollama running + model installed
        → LLM Mode → "show me CT reports" → expect results + AI response

Step 3  Verify AWS Bedrock
        USE_BEDROCK=true, fill AWS credentials
        → LLM Mode → "show me CT reports" → expect faster AI response

Step 4  Verify analytics
        → LLM Mode → "rank all radiologists by report count"
        → expect compact table (no raw records), ~5-10 second response

Step 5  Verify fuzzy matching
        → LLM Mode → "catscam reports in emergancy dept"
        → expect CT + EMERGENCY DEPT results

Step 6  Verify follow-up
        → LLM Mode → "show me MRI reports"
        → then "same but last month"
        → expect MRI reports filtered to last calendar month

Step 7  Add PostgreSQL (when ready)
        USE_PKL_DATA=false, fill POSTGRES_* vars
        → verify reports load from live database

Step 8  Add Azure AD (when deploying)
        DEV_MODE=false, fill AZURE_* vars
        → verify hospital Microsoft login works
```

---

## File Versions Quick Reference

When something breaks, replace these files in order:

| File | Location | What it does |
|---|---|---|
| `config.py` | project root | All settings — must have all 35 variables |
| `fuzzy.py` | project root | Fuzzy resolution — NEW file, must exist |
| `llm.py` | project root | LLM + analytics engine |
| `search.py` | project root | Search logic |
| `routes/llm_routes.py` | routes/ | LLM chat pipeline |
| `embeddings.py` | project root | ChromaDB management |

Always replace all of these together from the same set of outputs.
