# =============================================================================
# routes/llm_routes.py — LLM Chat Endpoints
# =============================================================================
#
# PIPELINE per user message:
#   0. Guard       — is LLM backend reachable?
#   1. Intent      — parse user message into structured params (ALL fields)
#   2. Scope check — is this something we can answer? If not, respond conversationally
#   3. Search      — fetch matching records using all available filters
#   4. Analytics   — if analytics/compare: run dynamic pandas code, get compact result
#   5. Context     — build AI context from records (full text or metadata depending on task)
#   6. Response    — stream AI answer back to browser
# =============================================================================

import json
import logging

from flask import Blueprint, request, jsonify, Response, stream_with_context

from auth import login_required
from search import run_search, build_table_html, get_filter_options
from llm import (
    ollama_chat,
    parse_intent,
    build_llm_context,
    execute_analytics,
    get_task_instruction,
)
from routes.api import reset_activity
from fuzzy import resolve_intent

llm_bp       = Blueprint("llm", __name__)
chat_sessions: dict = {}


@llm_bp.route("/llm_query", methods=["POST"])
@login_required
def llm_query():
    reset_activity()
    data       = request.get_json(silent=True) or {}
    user_msg   = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    history = chat_sessions[session_id]

    def generate():

        # ── STEP 0: Guard ─────────────────────────────────────────────────────
        from llm import ollama_available
        from config import USE_BEDROCK as _BEDROCK
        ok, models = ollama_available()

        if not ok:
            yield _sse({"type": "error", "text": (
                "AWS Bedrock is unreachable. Check credentials in .env and confirm "
                "Claude 3 Haiku has model access in ap-southeast-2."
                if _BEDROCK else
                "AI assistant is offline. Start Ollama with: ollama serve"
            )})
            yield _sse({"type": "done"}); return

        if not models:
            yield _sse({"type": "error", "text": (
                "Bedrock reachable but no model returned. Check BEDROCK_LLM_MODEL in .env."
                if _BEDROCK else
                "Ollama running but no models installed. Run: ollama pull phi3"
            )})
            yield _sse({"type": "done"}); return

        # ── STEP 1: Parse intent ──────────────────────────────────────────────
        yield _sse({"type": "status", "text": "🔍 Understanding your query…"})
        filter_opts = get_filter_options()

        try:
            intent = parse_intent(user_msg, filter_opts, history=history)
        except ValueError:
            logging.warning("[llm] Intent parse failed — using raw query fallback")
            intent = {
                "search_query": user_msg, "radiologist": None, "modality": None,
                "clinic": None, "exam_code": None, "exam_description": None,
                "start_date": None, "end_date": None,
                "start_datetime": None, "end_datetime": None,
                "top_n": None, "task": "search",
                "analytics_intent": None,
                "out_of_scope": False, "out_of_scope_reason": None,
            }
        except RuntimeError as e:
            yield _sse({"type": "error", "text": f"AI error during parsing: {e}"})
            yield _sse({"type": "done"}); return
        except Exception as e:
            yield _sse({"type": "error", "text": f"Unexpected error: {e}"})
            yield _sse({"type": "done"}); return

        logging.info(f"[llm] Raw intent: {intent}")

        # ── FUZZY RESOLUTION: normalise all values ──────────────────────────
        intent = resolve_intent(intent, user_msg, filter_opts)
        logging.info(f"[llm] Resolved intent: {intent}")

        # ── STEP 2: Out-of-scope check ────────────────────────────────────────
        if intent.get("out_of_scope"):
            reason = intent.get("out_of_scope_reason") or "This is outside what I can help with."
            logging.info(f"[llm] Out of scope: {reason}")

            oos_system = (
                "You are RadiSearch, an AI assistant for a hospital radiology database.\n\n"
                "WHAT YOU CAN DO:\n"
                "  - Search reports: by condition, modality, radiologist, clinic, exam type, date\n"
                "  - Summarise clinical findings across reports\n"
                "  - Answer questions about report content\n"
                "  - Analytics: counts, rankings, trends, breakdowns by any field\n"
                "    (by radiologist, modality, clinic, exam code, month, day, shift, etc.)\n"
                "  - Compare volumes across time periods, doctors, clinics, modalities\n"
                "  - Rank reports by clinical relevance\n\n"
                "WHAT YOU CANNOT DO:\n"
                "  - Access patient names, DOB, URN, contact details\n"
                "  - Create charts or graphs\n"
                "  - Send emails or schedule tasks\n"
                "  - Answer general knowledge questions\n\n"
                f"REASON OUT OF SCOPE: {reason}\n\n"
                "Respond conversationally in 2-3 sentences. "
                "If greeting: introduce yourself briefly and give 2 example queries. "
                "If off-topic: note your specialisation and suggest what you CAN help with. "
                "If PII: explain what data is available instead. "
                "Be friendly and helpful."
            )
            history.append({"role": "user", "content": user_msg})
            messages = [{"role": "system", "content": oos_system}] + history[-10:]

            yield _sse({"type": "status", "text": "💬 Responding…"})
            full = ""
            try:
                for chunk in ollama_chat(messages, stream=True):
                    full += chunk
                    yield _sse({"type": "token", "text": chunk})
            except Exception as e:
                full = f"I can't help with that — {reason}"
                yield _sse({"type": "token", "text": full})

            history.append({"role": "assistant", "content": full})
            chat_sessions[session_id] = history[-20:]
            yield _sse({"type": "done"}); return

        # ── STEP 3: Database search ───────────────────────────────────────────
        task             = intent.get("task", "search")
        analytics_intent = intent.get("analytics_intent") or user_msg

        # Enrich analytics_intent with resolved field values so the pandas
        # code generator uses the actual database values (e.g. CR not XR)
        resolved_notes = []
        if intent.get("modality"):
            resolved_notes.append(f"modality column value is '{intent['modality']}' (already filtered)")
        if intent.get("radiologist"):
            resolved_notes.append(f"radiologist is '{intent['radiologist']}' (already filtered)")
        if intent.get("clinic"):
            resolved_notes.append(f"clinic is '{intent['clinic']}' (already filtered)")
        if resolved_notes:
            analytics_intent = analytics_intent + ". NOTE: " + "; ".join(resolved_notes)
            analytics_intent += ". The records passed in are already filtered — do NOT re-filter by modality/radiologist/clinic in your code."
        raw_top_n        = intent.get("top_n")
        top_n            = int(raw_top_n) if raw_top_n is not None else None

        yield _sse({"type": "status", "text": "📂 Searching the database…"})

        records = run_search(
            search_query     = intent.get("search_query"),
            radiologist      = intent.get("radiologist"),
            modality         = intent.get("modality"),
            clinic           = intent.get("clinic"),
            start_date       = intent.get("start_date"),
            end_date         = intent.get("end_date"),
            top_n            = top_n,
            exam_code        = intent.get("exam_code"),
            exam_description = intent.get("exam_description"),
            start_datetime   = intent.get("start_datetime"),
            end_datetime     = intent.get("end_datetime"),
        )

        logging.info(f"[llm] Search → {len(records):,} records, task={task}")

        # For analytics/compare: do NOT show the raw records table.
        # The user wants a computed summary (e.g. radiologist counts), not
        # a list of 20,000 individual reports. The AI response IS the table.
        # For all other tasks: show the records table immediately so the user
        # sees results while the AI generates its response.
        task_for_table = intent.get("task", "search")
        if task_for_table not in ("analytics", "compare"):
            table_html = build_table_html(
                records,
                query            = intent.get("search_query"),
                result_id_prefix = "llm",
            )
            yield _sse({"type": "table", "html": table_html, "count": len(records), "intent": intent})
        else:
            # Still send the intent chips so the user sees what filters were applied
            yield _sse({"type": "table", "html": "", "count": len(records), "intent": intent})

        # Analytics fallback: if still empty, try all meta_data records
        if not records and task in ("analytics", "compare"):
            from embeddings import meta_data as _meta
            if _meta:
                logging.info(f"[llm] Analytics fallback → {len(_meta):,} records")
                records = list(_meta)

        # No records: explain why and suggest alternatives
        if not records:
            no_result_system = (
                "You are RadiSearch. A search returned zero results.\n\n"
                f"User asked: \"{user_msg}\"\n"
                f"Filters used: {json.dumps({k:v for k,v in intent.items() if v and k not in ('task','out_of_scope','out_of_scope_reason','analytics_intent')}, default=str)}\n\n"
                "In 2-3 sentences:\n"
                "1. Confirm no results matched\n"
                "2. Suggest why (modality code wrong, date too narrow, term not in reports, typo)\n"
                "3. Suggest a broader/corrected query to try"
            )
            history.append({"role": "user", "content": user_msg})
            full = ""
            yield _sse({"type": "status", "text": "🤖 Explaining no results…"})
            try:
                for chunk in ollama_chat(
                    [{"role": "system", "content": no_result_system}], stream=True
                ):
                    full += chunk
                    yield _sse({"type": "token", "text": chunk})
            except Exception:
                full = "No reports matched. Try removing date filters or using different keywords."
                yield _sse({"type": "token", "text": full})
            history.append({"role": "assistant", "content": full})
            chat_sessions[session_id] = history[-20:]
            yield _sse({"type": "done"}); return

        # ── STEP 4: Analytics / Compare ───────────────────────────────────────
        is_analytics = task in ("analytics", "compare")
        task_labels  = {
            "analytics": "⚙️ Computing analytics…",
            "compare":   "⚙️ Comparing groups…",
            "summarise": "📝 Summarising findings…",
            "answer":    "💡 Formulating answer…",
            "rank":      "🏆 Ranking reports…",
            "search":    "🤖 Generating overview…",
        }
        yield _sse({"type": "status", "text": task_labels.get(task, "🤖 Generating response…")})

        if is_analytics:
            # Dynamically generate and execute pandas code
            context_text, code_used = execute_analytics(user_msg, analytics_intent, records)
            logging.info(f"[llm] Analytics code: {code_used[:200]!r}")

            # Build filter context so AI knows exactly what data was used
            filter_summary = []
            if intent.get("modality"):
                filter_summary.append(f"modality={intent['modality']}")
            if intent.get("radiologist"):
                filter_summary.append(f"radiologist={intent['radiologist']}")
            if intent.get("clinic"):
                filter_summary.append(f"clinic={intent['clinic']}")
            if intent.get("start_date"):
                filter_summary.append(f"from {intent['start_date']}")
            if intent.get("end_date"):
                filter_summary.append(f"to {intent['end_date']}")
            filter_desc = (
                "Filters applied: " + ", ".join(filter_summary)
                if filter_summary else "No filters — full dataset"
            )

            system_msg = (
                "You are a clinical radiology analytics assistant.\n\n"
                "A Python/pandas query was dynamically generated and executed "
                "against the dataset. The result is shown below.\n\n"
                f"DATASET: {len(records):,} records | {filter_desc}\n\n"
                "IMPORTANT — if the user mentioned 'XR' or 'xray', this system "
                "stores X-Ray reports as modality='CR' (Computed Radiography). "
                "XR and CR are the same thing in this database. Do NOT say the "
                "data is missing or suggest checking field names.\n\n"
                "INSTRUCTIONS:\n"
                "- Present the result table clearly and completely\n"
                "- If result has a count/ranking column, sort it descending\n"
                "- Highlight the key insight in 2-3 sentences\n"
                "- If result returned no rows, explain that the filters may be "
                "too restrictive and suggest broadening the query\n"
                "- Never say you have limited access or suggest checking column names\n\n"
                f"COMPUTED RESULT:\n{context_text}\n\n"
                f"USER REQUEST: {user_msg}\n\n"
                f"YOUR TASK:\n{get_task_instruction(task, user_msg, len(records))}"
            )
        else:
            # ── STEP 5: Standard context ──────────────────────────────────────
            context_text = build_llm_context(records, task=task)
            system_msg = (
                "You are a clinical radiology assistant.\n\n"
                "You have the COMPLETE matched results — not a sample. "
                "Answer directly and completely. Never say you have limited access.\n\n"
                "AVAILABLE FIELDS: visit_number, report_date, visit_datetime, "
                "radiologist, modality, clinic, exam_code, exam_description, report_text\n"
                "NOT AVAILABLE: patient names, URNs, DOB, contact info, referring doctors\n\n"
                f"MATCHED RECORDS: {len(records):,} total\n"
                f"{context_text}\n\n"
                f"USER REQUEST: {user_msg}\n\n"
                f"YOUR TASK:\n{get_task_instruction(task, user_msg, len(records))}"
            )

        # ── STEP 6: Stream response ───────────────────────────────────────────
        history.append({"role": "user", "content": user_msg})
        messages = [{"role": "system", "content": system_msg}] + history[-10:]

        full = ""
        try:
            for chunk in ollama_chat(messages, stream=True):
                full += chunk
                yield _sse({"type": "token", "text": chunk})
        except Exception as e:
            err = f"\n\n[Error: {e}]"
            full += err
            logging.error(f"[llm] Stream error: {e}")
            yield _sse({"type": "token", "text": err})

        history.append({"role": "assistant", "content": full})
        chat_sessions[session_id] = history[-20:]
        yield _sse({"type": "done"})

    return Response(
        stream_with_context(generate()),
        mimetype = "text/event-stream",
        headers  = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@llm_bp.route("/clear_session", methods=["POST"])
@login_required
def clear_session():
    reset_activity()
    sid = (request.get_json(silent=True) or {}).get("session_id", "default")
    if sid in chat_sessions:
        del chat_sessions[sid]
        logging.info(f"[llm] Cleared session: {sid}")
    return jsonify({"ok": True})


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"