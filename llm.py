# =============================================================================
# llm.py — RadiSearch LLM Integration
# =============================================================================
#
# Handles all natural language intelligence for RadiSearch:
#   - Intent parsing:   understands ANY user query, handles typos/follow-ups
#   - Analytics engine: generates & executes pandas code dynamically
#   - Context builder:  formats records efficiently for each task type
#   - Task router:      gives the AI precise instructions per task
#
# Supports both local Ollama and AWS Bedrock transparently.
#
# DATA SCHEMA (from pkl / PostgreSQL):
#   visit_number     — hospital visit ID
#   patient_urn      — patient identifier (anonymised)
#   visit_datetime   — datetime of the visit (when the scan happened)
#   report_date      — date the report was written
#   radiologist      — full name of reporting radiologist
#   modality         — imaging type: CT, MRI, XA, CR, US, NM, DX, MG, RF, PT, OT
#   clinic           — clinic or department name
#   exam_code        — hospital exam/procedure code
#   exam_description — human-readable exam description
#   clean_report     — full cleaned text of the radiology report
# =============================================================================

import json
import re
import logging
import urllib.request
import urllib.error
from datetime import datetime

from config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_MODEL_PREFERENCE,
    USE_BEDROCK,
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BEDROCK_LLM_MODEL,
    BEDROCK_MAX_TOKENS,
)


# =============================================================================
# BEDROCK CLIENT
# =============================================================================

_bedrock_client = None

def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        _bedrock_client = boto3.Session(
            aws_access_key_id     = AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
            region_name           = AWS_REGION,
        ).client("bedrock-runtime")
        logging.info(f"[llm] Bedrock client created ({AWS_REGION}, {BEDROCK_LLM_MODEL})")
    return _bedrock_client


# =============================================================================
# AVAILABILITY & MODEL SELECTION
# =============================================================================

def ollama_available() -> tuple[bool, list]:
    return _bedrock_available() if USE_BEDROCK else _ollama_available()

def _ollama_available() -> tuple[bool, list]:
    try:
        with urllib.request.urlopen(
            urllib.request.Request("http://localhost:11434/api/tags"), timeout=3
        ) as resp:
            data = json.loads(resp.read())
            return True, [m["name"].split(":")[0] for m in data.get("models", [])]
    except Exception:
        return False, []

def _bedrock_available() -> tuple[bool, list]:
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        return False, []
    try:
        import boto3
        boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        ).client("bedrock").list_foundation_models(byOutputModality="TEXT")
        short = BEDROCK_LLM_MODEL.split(".")[1] if "." in BEDROCK_LLM_MODEL else BEDROCK_LLM_MODEL
        return True, [short]
    except Exception as e:
        logging.warning(f"[llm] Bedrock check failed: {e}")
        return False, []

def pick_ollama_model() -> str:
    if USE_BEDROCK:
        p = BEDROCK_LLM_MODEL.split(".")
        return p[1] if len(p) > 1 else BEDROCK_LLM_MODEL
    _, avail = _ollama_available()
    for pref in OLLAMA_MODEL_PREFERENCE:
        for inst in avail:
            if inst.lower().startswith(pref):
                return inst
    return avail[0] if avail else OLLAMA_MODEL


# =============================================================================
# CHAT
# =============================================================================

def ollama_chat(messages: list, stream: bool = False):
    return _bedrock_chat(messages, stream) if USE_BEDROCK else _ollama_chat(messages, stream)

def _ollama_chat(messages: list, stream: bool = False):
    payload = json.dumps({
        "model":    pick_ollama_model(),
        "messages": messages,
        "stream":   stream,
        "options":  {"temperature": 0.1, "num_predict": 4096},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    if stream:
        def _gen():
            try:
                with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as r:
                    for line in r:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            c = json.loads(line)
                            d = c.get("message", {}).get("content", "")
                            if d: yield d
                            if c.get("done"): break
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                yield f"\n\n[Ollama error: {e}]"
        return _gen()
    else:
        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as r:
                return json.loads(r.read()).get("message", {}).get("content", "")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Ollama ({e.reason})") from e
        except Exception as e:
            raise RuntimeError(f"Ollama failed: {e}") from e

def _bedrock_chat(messages: list, stream: bool = False):
    client  = _get_bedrock_client()
    system  = ""
    convo   = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            convo.append({"role": m["role"], "content": m["content"]})
    if not convo or convo[0]["role"] != "user":
        convo.insert(0, {"role": "user", "content": "Hello"})
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": BEDROCK_MAX_TOKENS,
        "messages":   convo,
        "temperature": 0.1,
    }
    if system:
        body["system"] = system
    return _bedrock_stream(client, body) if stream else _bedrock_invoke(client, body)

def _bedrock_invoke(client, body: dict) -> str:
    try:
        r = client.invoke_model(
            modelId=BEDROCK_LLM_MODEL, body=json.dumps(body),
            contentType="application/json", accept="application/json",
        )
        return json.loads(r["body"].read()).get("content", [{}])[0].get("text", "")
    except Exception as e:
        raise RuntimeError(f"Bedrock invoke failed: {e}") from e

def _bedrock_stream(client, body: dict):
    try:
        r = client.invoke_model_with_response_stream(
            modelId=BEDROCK_LLM_MODEL, body=json.dumps(body),
            contentType="application/json", accept="application/json",
        )
        s = r.get("body")
        if not s:
            yield "[No response from Bedrock]"; return
        for event in s:
            chunk = event.get("chunk")
            if not chunk: continue
            try:
                d = json.loads(chunk["bytes"])
                if d.get("type") == "content_block_delta":
                    t = d.get("delta", {}).get("text", "")
                    if t: yield t
                elif d.get("type") == "message_stop":
                    break
            except (json.JSONDecodeError, KeyError):
                continue
    except Exception as e:
        logging.error(f"[llm] Bedrock stream error: {e}")
        yield f"\n\n[Bedrock error: {e}]"


# =============================================================================
# INTENT SYSTEM PROMPT
# =============================================================================

_INTENT_SYSTEM_TEMPLATE = """You are an intelligent query parser for a hospital radiology report database.
Understand what the user wants — no matter how they phrase it.
Handle typos, casual language, abbreviations, and follow-up questions.

Today is {today}.

════════════════════════════════════════════════
FULL DATABASE SCHEMA
════════════════════════════════════════════════
  visit_number     — hospital visit ID (numeric string)
  patient_urn      — anonymised patient identifier (NOT a name)
  visit_datetime   — datetime the scan/visit occurred (datetime64)
  report_date      — date the report was written (date)
  radiologist      — full name of the reporting radiologist (str)
  modality         — imaging type code (str): CT, MRI, XA, CR, US, NM, DX, MG, RF, PT, OT
  clinic           — clinic or department name (str)
  exam_code        — hospital exam/procedure code (str)
  exam_description — human-readable exam name e.g. "CT CHEST WITH CONTRAST" (str)
  clean_report     — full text of the radiology report (findings, impressions, etc.)

NOT in database: patient names, DOB, age, address, phone, referring doctor,
GP name, ICD codes, billing codes, appointment slots.

════════════════════════════════════════════════
AVAILABLE VALUES (fuzzy-match user input to these)
════════════════════════════════════════════════
Radiologists : {radiologists}
Modalities   : {modalities}
Clinics      : {clinics}

════════════════════════════════════════════════
CONVERSATION HISTORY
════════════════════════════════════════════════
{history}

════════════════════════════════════════════════
OUTPUT — respond ONLY with this JSON
════════════════════════════════════════════════
{{
  "search_query":        <string|null>,
  "radiologist":         <string|null>,
  "modality":            <string|null>,
  "clinic":              <string|null>,
  "exam_code":           <string|null>,
  "exam_description":    <string|null>,
  "start_date":          <"YYYY-MM-DD"|null>,
  "end_date":            <"YYYY-MM-DD"|null>,
  "start_datetime":      <"YYYY-MM-DD HH:MM"|null>,
  "end_datetime":        <"YYYY-MM-DD HH:MM"|null>,
  "top_n":               <integer|null>,
  "task":                <"search"|"summarise"|"answer"|"analytics"|"rank"|"compare">,
  "analytics_intent":    <string|null>,
  "out_of_scope":        <true|false>,
  "out_of_scope_reason": <string|null>
}}

════════════════════════════════════════════════
TASK CLASSIFICATION
════════════════════════════════════════════════
"search"    → browse/retrieve matching reports
              Triggers: show, find, get, give me, list, display, fetch
              search_query = clinical keywords

"summarise" → written clinical summary of findings
              Triggers: summarise, summary, overview, what findings, describe,
                        what do reports say, common diagnoses, patterns in
              search_query = clinical topic

"answer"    → specific question about report content
              Triggers: what is, which, how, tell me, explain, is there,
                        what conditions, what were findings, how many reports
                        mention X (where X is a clinical term in report text)
              search_query = clinical topic

"analytics" → counts/rankings/trends/stats about the DATA ITSELF
              Triggers: rank, how many, busiest, most, count, volume,
                        breakdown, performance, trend, per month, per week,
                        per day, per year, which doctor did most, top N,
                        frequency, distribution, workload, throughput,
                        reports done, exam codes, common exams
              analytics_intent = concise description for pandas code generator
              search_query = null UNLESS filtering by clinical condition
                             e.g. "rank doctors by cancer reports" → search_query="cancer"

"rank"      → rank REPORTS by clinical relevance to a topic
              Triggers: rank reports, most relevant, prioritise, order by relevance
              search_query = clinical topic

"compare"   → compare two or more groups, periods, doctors, or clinics
              Triggers: compare, vs, versus, difference, contrast, how does X compare
              analytics_intent = what is being compared

════════════════════════════════════════════════
ANALYTICS INTENT — fill when task=analytics or compare
════════════════════════════════════════════════
A plain English description of what the pandas code should compute.
Be specific so the code generator knows exactly what to produce. Examples:

"rank radiologists by reports done every month" →
  analytics_intent: "group by radiologist and month, count reports, pivot month as columns"

"busiest clinic" →
  analytics_intent: "group by clinic, count reports, sort descending"

"CT vs MRI volume by clinic" →
  analytics_intent: "filter to CT and MRI, group by clinic and modality, count, pivot"

"morning vs afternoon reports" →
  analytics_intent: "extract hour from visit_datetime, group AM (<12) vs PM (>=12), count"

"top 5 exam codes" →
  analytics_intent: "group by exam_code and exam_description, count, sort desc, head 5"

"how many reports per day last week" →
  analytics_intent: "filter to last 7 days by report_date, group by date, count"

"radiologist performance per modality" →
  analytics_intent: "group by radiologist and modality, count, pivot modality as columns"

════════════════════════════════════════════════
FIELD SELECTION RULES
════════════════════════════════════════════════
search_query:
  - Clinical terms, diagnoses, body regions, findings, conditions
  - For analytics/compare with no clinical filter: null
  - "all reports" / "everything": null
  - Keep to 1-5 clinical words

exam_code / exam_description:
  - Set when user mentions specific exam type by code or name
  - e.g. "XR CHEST" → exam_description="XR CHEST", or exam_code if code given

start_datetime / end_datetime:
  - Use instead of start_date/end_date when user mentions time of day
  - "morning reports" → start_datetime="YYYY-MM-DD 00:00", end_datetime="YYYY-MM-DD 11:59"
  - "afternoon" → 12:00 to 17:59, "evening" → 18:00 to 23:59

top_n:
  - null by default (ALL records)
  - Set ONLY when user says: "top 10", "give me 5", "show 50", "first 20"
  - "all", "every", "complete", "full" → null

DATE RULES (only when user explicitly mentions a period):
  last month     → first/last day of previous calendar month
  this month     → 1st of current month to today
  this week      → Monday of current week to today
  last 7 days    → today-7 to today
  last 30 days   → today-30 to today
  last year      → Jan 1 to Dec 31 of previous year
  this year      → Jan 1 of current year to today
  yesterday      → yesterday
  today          → today
  in 2024        → 2024-01-01 to 2024-12-31
  Q1 2025        → 2025-01-01 to 2025-03-31
  Q2             → Apr 1 to Jun 30 (current year)
  last quarter   → previous Q dates
  Never invent dates. Never carry over dates from history.

════════════════════════════════════════════════
SPELLING / TYPO RULES
════════════════════════════════════════════════
Match user input to available values even with errors:
  catscam / cat scan / ctscan → CT
  mri scan / mrI               → MRI
  xray / x-ray / x ray         → XR or CR
  ultrasound / ultra sound     → US
  mammogram / mammo             → MG
  nuclear / nuclear med         → NM
  Radiologist name typos       → closest match in available list
  Clinic name typos            → closest match in available list
  emergancy / emergeny         → EMERGENCY DEPT (or closest)

════════════════════════════════════════════════
FOLLOW-UP RULES
════════════════════════════════════════════════
Resolve follow-ups using conversation history:
  "now filter by CT"         → add modality=CT, keep rest
  "same but last month"      → update dates, keep all other params
  "all radiologists"         → remove radiologist filter, keep rest
  "top 10 only"              → set top_n=10, keep rest
  "what about MRI instead"   → change modality=MRI, keep rest
  "now summarise them"       → change task=summarise, keep params
  "who was busiest"          → analytics task on same dataset
  "same for Emergency Dept"  → set clinic=EMERGENCY DEPT
  "now show Dr Smith only"   → set radiologist=Dr Smith
  Never carry over dates unless user says "same period".

════════════════════════════════════════════════
out_of_scope — true ONLY for:
════════════════════════════════════════════════
  1. Patient PII: names, DOB, URN, MRN, phone, address
  2. Visualisations: charts, graphs, plots, diagrams
  3. External actions: email, alert, schedule, export PDF, book appointment
  4. Pure general knowledge: weather, news, sports, jokes, maths
  5. Zero-intent greetings: "hello", "hi", "thanks", "bye"

Set false for EVERYTHING related to the report data.

════════════════════════════════════════════════
EXAMPLES
════════════════════════════════════════════════
"rank all radiologists by number of reports done"
→ task:analytics, analytics_intent:"group by radiologist, count reports, sort desc", search_query:null

"rank doctors by reports done every month"
→ task:analytics, analytics_intent:"group by radiologist and month (YYYY-MM), count, pivot month as columns, sort by total desc"

"who is the busiest radiologist?"
→ task:analytics, analytics_intent:"group by radiologist, count reports, sort desc, head 1"

"compare CT and MRI volumes by clinic"
→ task:compare, analytics_intent:"filter CT and MRI, group by clinic and modality, count, pivot modality"

"show me all pleural effusion reports"
→ task:search, search_query:"pleural effusion"

"how many reports mention cancer this year?"
→ task:analytics, search_query:"cancer", analytics_intent:"count total records", start_date:Jan 1 this year

"summarise CT chest findings"
→ task:summarise, search_query:"chest", modality:CT

"which exam codes are most common?"
→ task:analytics, analytics_intent:"group by exam_code and exam_description, count, sort desc"

"morning shift report volumes"
→ task:analytics, analytics_intent:"extract hour from visit_datetime, group AM (<12) vs PM (>=12), count by radiologist"

"give me top 5 fracture reports"
→ task:rank, search_query:"fracture", top_n:5

"smyths reports last week" (typo)
→ task:search, radiologist:closest SMITH match, dates:last 7 days

"whos the best doctor"
→ task:analytics, analytics_intent:"group by radiologist, count reports, sort desc — highest volume as proxy for busiest"

"catscam reports in emergancy" (typos)
→ task:search, modality:CT, clinic:closest EMERGENCY match

"compare this month vs last month"
→ task:compare, analytics_intent:"group by month (this month vs last month), count reports, show side by side"

"now rank by relevance" (follow-up)
→ task:rank, carry forward search_query from previous turn

"create a chart"
→ out_of_scope:true

"what is the weather"
→ out_of_scope:true

"give me all patient names"
→ out_of_scope:true
"""


# =============================================================================
# INTENT PARSER
# =============================================================================

def parse_intent(user_query: str, filter_options: dict, history: list = None) -> dict:
    """
    Converts any natural language query into structured search parameters.
    Handles typos, follow-ups, casual phrasing, all task types, all fields.
    """
    today = datetime.today()

    history_text = "(no previous messages)"
    if history:
        lines = []
        for msg in history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            text = msg["content"][:400] + ("..." if len(msg["content"]) > 400 else "")
            lines.append(f"{role}: {text}")
        history_text = "\n".join(lines)

    system_prompt = _INTENT_SYSTEM_TEMPLATE.format(
        today        = today.strftime("%Y-%m-%d"),
        radiologists = ", ".join(filter_options.get("radiologists", [])[:40]),
        modalities   = ", ".join(filter_options.get("modalities",   [])),
        clinics      = ", ".join(filter_options.get("clinics",      [])[:30]),
        history      = history_text,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query},
    ]

    raw = ""
    try:
        raw     = ollama_chat(messages, stream=False)
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        parsed = json.loads(cleaned)
        logging.info(f"[llm] Intent: {parsed}")
        return parsed
    except json.JSONDecodeError as e:
        logging.error(f"[llm] Intent JSON failed: {e} | raw: {raw[:300]!r}")
        raise ValueError("AI returned unexpected format")
    except RuntimeError:
        raise
    except Exception as e:
        logging.error(f"[llm] Intent error ({type(e).__name__}): {e}")
        raise RuntimeError(f"LLM failed: {e}") from e


# =============================================================================
# ANALYTICS ENGINE
# =============================================================================
#
# Two-step process:
#   1. Ask the LLM to write a pandas snippet for the specific request
#   2. Execute it safely on the full DataFrame
#
# Handles any grouping/aggregation dynamically:
#   - group by one field, multiple fields, or derived fields (month, hour, year)
#   - filter then group
#   - pivot tables
#   - percentages, top-N within groups
#   - time-based trends (daily, weekly, monthly, quarterly, yearly)
#   - cross-tabulations
#   - running totals
#   - comparisons between periods or groups
# =============================================================================

_ANALYTICS_CODE_PROMPT = """You are a Python/pandas expert writing code for a radiology report DataFrame.

DataFrame: df  (already filtered to the relevant subset — do NOT re-filter by modality/radiologist/clinic)
Columns:
  radiologist      str   — radiologist full name
  modality         str   — actual values in this dataset: CT, MRI, XA, CR, US, NM, DX, MG, RF, PT, OT
                           NOTE: X-Ray is stored as "CR" (Computed Radiography) — NOT "XR" or "XRAY"
  clinic           str   — clinic/department name
  exam_code        str   — exam/procedure code
  exam_description str   — exam name e.g. "CT CHEST WITH CONTRAST"
  visit_datetime   datetime64 — when the scan occurred
  report_date      datetime64 — when report was written
  visit_number     str   — visit ID

USER REQUEST: {user_msg}
ANALYTICS INTENT: {analytics_intent}

CRITICAL: The DataFrame is already pre-filtered. Do NOT add where/filter conditions
for modality, radiologist, or clinic unless the user is explicitly comparing multiple
values (e.g. "CT vs MRI"). Just group, count, and sort the data as-is.

Write Python code assigning `result` to a DataFrame answering the request.

ALLOWED: pandas (pd), numpy (np), built-in Python only.
RULES:
  - result must always be a DataFrame
  - For scalar results: pd.DataFrame({{"value": [the_scalar]}})
  - For groupby counts: df.groupby([...]).size().reset_index(name="count")
  - For month: df["report_date"].dt.to_period("M").astype(str)  or  .dt.strftime("%Y-%m")
  - For year:  df["report_date"].dt.year
  - For quarter: df["report_date"].dt.to_period("Q").astype(str)
  - For hour:  df["visit_datetime"].dt.hour
  - For day of week: df["visit_datetime"].dt.day_name()
  - For pivots: pd.crosstab(...) or pd.pivot_table(...)
  - For top-N within groups: df.groupby(...)...groupby(level=0).nlargest(N)
  - For percentages: add col = (count / count.sum() * 100).round(1)
  - Sort meaningfully: .sort_values("count", ascending=False)
  - Column names should be clean, readable strings
  - Handle NaT/NaN: use .dropna() or .fillna("Unknown") where appropriate
  - NO print(), display(), file I/O, network, eval(), exec(), __import__
  - Cap output at 1000 rows if result would be very large

RESPOND WITH ONLY THE PYTHON CODE. No explanation, no markdown fences.

Examples:
# Rank radiologists by total reports
result = df.groupby("radiologist").size().reset_index(name="report_count").sort_values("report_count", ascending=False)

# Reports per radiologist per month
tmp = df.copy()
tmp["month"] = tmp["report_date"].dt.strftime("%Y-%m")
result = tmp.groupby(["radiologist","month"]).size().reset_index(name="count").sort_values(["radiologist","month"])

# CT vs MRI by clinic
result = pd.crosstab(df[df["modality"].isin(["CT","MRI"])]["clinic"], df[df["modality"].isin(["CT","MRI"])]["modality"]).reset_index()

# Top exam codes
result = df.groupby(["exam_code","exam_description"]).size().reset_index(name="count").sort_values("count", ascending=False).head(20)

# Morning vs afternoon by radiologist
tmp = df.copy()
tmp["shift"] = tmp["visit_datetime"].dt.hour.apply(lambda h: "AM" if h < 12 else "PM")
result = pd.crosstab(tmp["radiologist"], tmp["shift"]).reset_index()

# Monthly trend with percentage change
tmp = df.copy()
tmp["month"] = tmp["report_date"].dt.strftime("%Y-%m")
result = tmp.groupby("month").size().reset_index(name="count").sort_values("month")
result["pct_change"] = result["count"].pct_change().mul(100).round(1)
"""


def execute_analytics(user_msg: str, analytics_intent: str, records: list) -> tuple[str, str]:
    """
    Dynamically generates and executes a pandas query for any analytics request.

    Returns (result_markdown, code_used).
    Falls back gracefully if code generation or execution fails.
    """
    import pandas as pd
    import numpy as np

    if not records:
        return "(no records to analyse)", ""

    # Build DataFrame — include all fields, drop heavy report text
    df = pd.DataFrame(records)
    for col in ["report_date", "visit_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "clean_report" in df.columns:
        df = df.drop(columns=["clean_report"])

    total = len(df)

    # Step 1: Generate pandas code
    code_prompt = _ANALYTICS_CODE_PROMPT.format(
        user_msg         = user_msg,
        analytics_intent = analytics_intent or user_msg,
    )
    try:
        raw_code = ollama_chat(
            [{"role": "system", "content": code_prompt},
             {"role": "user",   "content": user_msg}],
            stream=False,
        )
        logging.info(f"[analytics] Code: {raw_code[:300]!r}")
    except Exception as e:
        logging.error(f"[analytics] Code generation failed: {e}")
        return _simple_fallback(df), ""

    code = re.sub(r"```(?:python)?|```", "", raw_code).strip()

    # Step 2: Execute safely
    safe_builtins = {
        "len": len, "range": range, "list": list, "dict": dict, "set": set,
        "str": str, "int": int, "float": float, "bool": bool, "tuple": tuple,
        "round": round, "min": min, "max": max, "sum": sum, "abs": abs,
        "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map,
        "filter": filter, "any": any, "all": all,
        "isinstance": isinstance, "type": type,
        "print": lambda *a, **k: None,  # no-op
        "None": None, "True": True, "False": False,
    }
    safe_globals = {"__builtins__": safe_builtins, "pd": pd, "np": np}
    local_vars   = {"df": df, "result": None}

    try:
        exec(code, safe_globals, local_vars)
        result = local_vars.get("result")
    except Exception as e:
        logging.warning(f"[analytics] Execution failed: {e} | code: {code[:300]}")
        return _simple_fallback(df), code

    # Step 3: Format result
    if result is None:
        return _simple_fallback(df), code

    try:
        if isinstance(result, pd.Series):
            result = result.reset_index()
            result.columns = [str(c) for c in result.columns]

        if not isinstance(result, pd.DataFrame):
            return f"**Result:** {result}\n*(from {total:,} total records)*", code

        if result.empty:
            return f"(query returned no rows from {total:,} records)", code

        # Cap at 500 rows for readability
        display = result.head(500)
        note    = f", showing top 500 of {len(result):,}" if len(result) > 500 else f"{len(result):,} rows"

        try:
            md = display.to_markdown(index=False)
        except Exception:
            md = display.to_string(index=False)

        return (
            f"**Total records analysed: {total:,}**\n\n"
            f"**Result ({note}):**\n\n{md}"
        ), code

    except Exception as e:
        logging.error(f"[analytics] Formatting failed: {e}")
        return _simple_fallback(df), code


def _simple_fallback(df) -> str:
    """Minimal fallback when dynamic code fails — produces basic counts."""
    import pandas as pd
    total = len(df)
    lines = [f"**Total records: {total:,}**\n"]

    for field, label in [("radiologist","By radiologist"), ("modality","By modality"),
                          ("clinic","By clinic"), ("exam_code","By exam code")]:
        if field not in df.columns:
            continue
        try:
            vc = df[field].value_counts().head(30).reset_index()
            vc.columns = [field, "count"]
            lines.append(f"\n**{label}:**")
            lines.append(vc.to_markdown(index=False))
        except Exception:
            pass

    try:
        if "report_date" in df.columns:
            df2 = df.copy()
            df2["month"] = df2["report_date"].dt.strftime("%Y-%m")
            mc = df2.groupby("month").size().reset_index(name="count").sort_values("month")
            lines.append("\n**By month:**")
            lines.append(mc.to_markdown(index=False))
    except Exception:
        pass

    return "\n".join(lines)


# =============================================================================
# CONTEXT BUILDER
# =============================================================================

_FULL_TOKENS  = 250   # metadata + 800-char report text per record
_TOKEN_BUDGET = 80_000


def build_llm_context(records: list, task: str = "search") -> str:
    """
    Builds the AI context string based on task type.

    analytics/compare → handled separately via execute_analytics (not called here)
    answer            → metadata rows for all records (AI counts from fields)
    summarise         → full text for top 50, metadata for rest
    rank              → full text for top 30
    search            → full text for top 15
    """
    if not records:
        return "(no matching records)"

    # answer: metadata-only for the full set — AI reads structured fields
    if task == "answer":
        return _metadata_rows(records)

    # text tasks: send report text for top N
    text_limit = {"summarise": 50, "rank": 30, "search": 15}.get(task, 15)
    lines = []
    used  = 0

    for i, r in enumerate(records, start=1):
        if i > text_limit or used + _FULL_TOKENS > _TOKEN_BUDGET:
            break
        d = r.get("report_date", "")
        if hasattr(d, "strftime"):
            d = d.strftime("%Y-%m-%d")
        lines.append(
            f"[{i}] {d} | {r.get('radiologist','')} | "
            f"{r.get('modality','')} | {r.get('clinic','')} | "
            f"{r.get('exam_description','')}\n"
            + str(r.get("clean_report",""))[:800]
        )
        used += _FULL_TOKENS

    return "\n\n".join(lines)


def _metadata_rows(records: list) -> str:
    """Compact metadata-only rows for answer tasks."""
    rows = []
    for i, r in enumerate(records, start=1):
        d = r.get("report_date", "")
        if hasattr(d, "strftime"):
            d = d.strftime("%Y-%m-%d")
        rows.append(
            f"[{i}] {d} | {r.get('radiologist','')} | "
            f"{r.get('modality','')} | {r.get('clinic','')} | "
            f"{r.get('exam_code','')} {r.get('exam_description','')}"
        )
        # Budget: ~30 tokens per row, stay under 120k
        if i * 30 > 120_000:
            rows.append(f"[... {len(records)-i} more records]")
            break
    return "\n".join(rows)


# =============================================================================
# TASK INSTRUCTION BUILDER
# =============================================================================

def get_task_instruction(task: str, user_msg: str, record_count: int) -> str:
    """
    Returns precise AI instructions for each task type.
    Covers every scenario a user could ask about the radiology data.
    """

    if task == "analytics":
        return (
            f'Fulfil this analytics request: "{user_msg}"\n\n'
            "A Python/pandas query was executed against the COMPLETE dataset.\n"
            "The computed result is shown above.\n\n"
            "Instructions:\n"
            "- Present the result table clearly\n"
            "- Highlight the key finding (top performer, outlier, trend direction)\n"
            "- Add 2-3 sentences of insight\n"
            "- If result seems incomplete, explain what was computed and suggest "
            "a refined query\n"
            "- Never say you have limited access — the full dataset was used"
        )

    elif task == "compare":
        return (
            f'Compare as requested: "{user_msg}"\n\n'
            "The computed comparison data is shown above.\n"
            "- Present the comparison clearly with the result table\n"
            "- Highlight meaningful differences and similarities\n"
            "- State which group is larger/smaller/faster/busier\n"
            "- End with a 2-3 sentence conclusion"
        )

    elif task == "summarise":
        return (
            f'Write a clinical summary for: "{user_msg}"\n\n'
            f"You have {record_count} reports. From the report texts:\n"
            "- Identify the most common findings, diagnoses, and clinical patterns\n"
            "- Group by body region, condition type, or modality where useful\n"
            "- Note unusual or significant findings\n"
            "- Use proper medical terminology\n"
            "- Use bullet points for readability\n"
            "- Reference the total number of reports"
        )

    elif task == "answer":
        return (
            f'Answer this question: "{user_msg}"\n\n'
            f"You have metadata for all {record_count} matched records.\n"
            "Each row has: date, radiologist, modality, clinic, exam code/description.\n"
            "- Answer directly and completely\n"
            "- Count, list, or group from the metadata rows as needed\n"
            "- Cite report numbers [N] where relevant\n"
            "- If reading report text is needed, use the full-text records provided\n"
            "- Never say you have limited access"
        )

    elif task == "rank":
        return (
            f'Rank reports by clinical relevance to: "{user_msg}"\n\n'
            "- Rank from most to least relevant\n"
            "- For each report: state its number [N] and one sentence on WHY it is relevant\n"
            "- Focus on clinical significance, not just keyword presence\n"
            f"- Show top 20 of the {record_count} matched reports"
        )

    else:  # search / default
        return (
            f'The user searched for: "{user_msg}"\n'
            f"{record_count} reports matched (table shown above).\n\n"
            "Write 2-4 sentences describing what was found:\n"
            "- Main clinical conditions or findings present\n"
            "- Date range covered\n"
            "- Which modalities and clinics appear most\n"
            "- Any notable patterns\n"
            "Keep it concise — the user can read individual reports in the table."
        )