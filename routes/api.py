# =============================================================================
# routes/api.py — JSON Utility API Endpoints
# =============================================================================
#
# WHAT THIS MODULE DOES:
#   Provides lightweight JSON endpoints that the browser calls in the
#   background to power the UI — dropdown menus, countdown timers,
#   search history pills, Ollama status indicators, and dashboard charts.
#
#   All routes in this file return JSON (via jsonify), never full HTML pages.
#   HTML pages are in routes/pages.py. Search results are in search_routes.py.
#
#   Route              Method  Returns                         Used by
#   ─────────────────────────────────────────────────────────────────────────
#   /heartbeat         GET     {remaining, expired}            All pages (timer)
#   /ping              POST    {ok}                            All pages (timer reset)
#   /progress          GET     {current, total, status}        Loading overlay
#   /filter_options    GET     {radiologists, modalities, clinics} Dropdowns
#   /history           GET     [query, query, ...]             Search history pills
#   /ollama_status     GET     {available, models, active_model} LLM topbar dot
#   /stats             GET     {totals, counts, trend}         Dashboard charts
#
# All routes are protected with @login_required except /heartbeat and /ping,
# which must remain accessible during session transitions so the countdown
# timer keeps working correctly even as sessions expire.
#
# BLUEPRINT:
#   api_bp — registered in app.py with app.register_blueprint(api_bp)
# =============================================================================

import os
import time
import threading
import logging
from datetime import datetime

import pandas as pd
from flask import Blueprint, jsonify, request, session

from auth import login_required
from config import INACTIVITY_LIMIT
from llm import ollama_available, pick_ollama_model

# Module-level shared state — imported from embeddings at request time
# to always get the current values (they change after a rebuild)
import embeddings as _emb

api_bp = Blueprint("api", __name__)

# =============================================================================
# INACTIVITY TIMER STATE
# =============================================================================
# last_activity is owned here in api.py because /ping (the reset endpoint)
# and /heartbeat (the read endpoint) both live here.
# pages.py and search_routes.py import reset_activity() to update it when
# a search or filter action is performed.

_last_activity: float = time.time()   # Initialised to app launch time

# Flag — set to True once embeddings finish loading.
# The inactivity watcher checks this before counting down so the app
# never auto-closes while the FAISS index is still being built.
_embeddings_ready: bool = False


def set_embeddings_ready() -> None:
    """
    Called by embeddings.py once the FAISS index is fully loaded or built.
    Activates the inactivity timer — it will not count down before this point.
    Also resets last_activity to now so the full 5 minutes starts fresh
    from the moment the app is actually ready to use.
    """
    global _embeddings_ready, _last_activity
    _embeddings_ready = True
    _last_activity    = time.time()
    logging.info("[api] Embeddings ready — inactivity timer activated.")


def is_embeddings_ready() -> bool:
    """Returns True once embeddings have finished loading."""
    return _embeddings_ready


def reset_activity() -> None:
    """
    Resets the inactivity timer to now.
    Called by search_routes.py and llm_routes.py whenever a real user
    action (search, filter, LLM query) is performed.

    Also called by the /ping route whenever the user clicks any button
    in the UI (search, reset, export, tab switch, etc.)
    """
    global _last_activity
    _last_activity = time.time()


def get_last_activity() -> float:
    """Returns the timestamp of the last recorded user activity."""
    return _last_activity


# =============================================================================
# ROUTES
# =============================================================================

@api_bp.route("/heartbeat")
def heartbeat():
    """
    Polled every second by every page in the app via JavaScript.

    Returns how many seconds remain before the inactivity watcher shuts
    the server down. The browser uses this to:
      - Update the countdown display in the topbar (e.g. "⏱ 04:32")
      - Redirect to /closed when remaining hits 0

    While embeddings are still loading, returns loading=true so the
    browser shows "Loading…" instead of a countdown — and critically,
    the inactivity watcher does not run during this period.

    NOT protected by @login_required — must work even if the session has
    just expired, so the browser can redirect cleanly to /closed rather
    than being bounced to /auth/login first.
    """
    if not _embeddings_ready:
        # Embeddings still building — suppress the timer entirely
        return jsonify({
            "remaining": INACTIVITY_LIMIT,
            "expired":   False,
            "loading":   True,
        }), 200, {"Cache-Control": "no-cache"}

    elapsed   = time.time() - _last_activity
    remaining = max(0, INACTIVITY_LIMIT - elapsed)
    return jsonify({
        "remaining": int(remaining),
        "expired":   remaining <= 0,
        "loading":   False,
    }), 200, {"Cache-Control": "no-cache"}


@api_bp.route("/ping", methods=["POST"])
def ping():
    """
    Resets the inactivity timer. Called by the browser on every user
    interaction — button clicks, keyboard Enter, tab switches, etc.

    The JavaScript ping() function on every page fires this as a
    fire-and-forget POST (errors are silently swallowed on the client).

    NOT protected by @login_required for the same reason as /heartbeat —
    the ping should work even during session transitions.

    Returns {"ok": true} with Cache-Control: no-cache to prevent
    any proxy or browser from caching the reset.
    """
    reset_activity()
    logging.debug("[api] Activity ping received — timer reset.")
    return jsonify({"ok": True}), 200, {"Cache-Control": "no-cache"}


@api_bp.route("/progress")
@login_required
def get_progress():
    """
    Returns the current embedding build progress as JSON.

    The browser's loading overlay polls this every 500ms while the
    FAISS index is being built in the background thread. It uses
    current/total to calculate a percentage for the progress bar,
    and status="done" to know when to hide the overlay.

    Example response:
        {"current": 320, "total": 500, "status": "loading"}
        {"current": 500, "total": 500, "status": "done"}

    Cache-Control: no-cache prevents the browser from serving stale
    progress values — without this, the progress bar can get stuck.
    """
    return jsonify(_emb.progress), 200, {"Cache-Control": "no-cache"}


@api_bp.route("/filter_options")
@login_required
def filter_options():
    """
    Returns the lists of unique radiologists, modalities, and clinics
    from the currently loaded dataset.

    Called once by the browser after the loading overlay disappears,
    to populate the dropdown menus in the Radiologist Report Metrics panel
    and the modality/clinic dropdowns.

    Also used by the LLM intent parser (via get_filter_options() in search.py)
    to give the AI the exact strings it needs to match names correctly.

    Example response:
        {
            "radiologists": ["Dr Adams", "Dr Brown", "Dr Smith"],
            "modalities":   ["CT", "MRI", "X-Ray"],
            "clinics":      ["City Clinic", "North Campus"]
        }
    """
    from search import get_filter_options
    return jsonify(get_filter_options())


@api_bp.route("/history")
@login_required
def get_history():
    """
    Returns the last 20 search queries typed in Custom Search mode.

    Displayed as clickable "pill" buttons under the search box so the
    user can quickly repeat a recent query without retyping it.

    History is stored in memory (search_history list in search_routes.py)
    and cleared when the app restarts. It is not persisted to disk.

    Example response:
        ["cancer", "pleural effusion", "fracture left femur"]
    """
    from routes.search_routes import search_history
    return jsonify(search_history[-20:])


@api_bp.route("/ollama_status")
@login_required
def ollama_status():
    """
    Returns whether Ollama is running and which models are installed.

    Used by the LLM mode page to:
      - Show a green dot (online) or red dot (offline) in the topbar
      - Show the active model name (e.g. "phi3") next to the dot
      - Show/hide the offline warning banner

    Also called at page load so the status is shown immediately without
    waiting for the user to try sending a message.

    Example responses:
        {"available": true,  "models": ["phi3", "mistral"], "active_model": "phi3"}
        {"available": false, "models": [],                  "active_model": null}
    """
    ok, models = ollama_available()
    no_models  = ok and len(models) == 0
    return jsonify({
        "available":    ok and len(models) > 0,
        "no_models":    no_models,
        "models":       models,
        "active_model": pick_ollama_model() if (ok and models) else None,
    })


@api_bp.route("/stats")
@login_required
def get_stats():
    """
    Returns summary statistics about the report database for the Dashboard tab.

    Called when the user switches to the Dashboard tab (lazy-loaded —
    not fetched on page load to keep initial load fast).

    Returns:
        total_reports    — total records in the loaded dataset
        today_reports    — records with report_date == today
        modality_counts  — {"CT": 120, "MRI": 85, "X-Ray": 200, ...}
        top_radiologists — top 10 by report count {"Dr Smith": 45, ...}
        daily_trend      — {"2025-03-01": 12, "2025-03-02": 18, ...}
                           last 30 days, used for the sparkline bar chart

    All data comes from the in-memory meta_data list loaded at startup —
    no database query is made at request time.
    """
    if _emb.meta_data is None:
        from embeddings import initialize_embeddings
        initialize_embeddings()

    df    = pd.DataFrame(_emb.meta_data)
    today = datetime.today().date()

    # ── Today's report count ─────────────────────────────────────────────────
    if "report_date" in df.columns:
        today_count = int((df["report_date"].dt.date == today).sum())
    else:
        today_count = 0

    # ── Modality breakdown ────────────────────────────────────────────────────
    modality_counts = (
        df["modality"].value_counts().to_dict()
        if "modality" in df.columns else {}
    )

    # ── Top 10 radiologists ───────────────────────────────────────────────────
    rad_counts = (
        df["radiologist"].value_counts().head(10).to_dict()
        if "radiologist" in df.columns else {}
    )

    # ── 30-day trend (daily report counts) ───────────────────────────────────
    trend = {}
    if "report_date" in df.columns:
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=29)
        recent = df[df["report_date"] >= cutoff]
        trend  = (
            recent
            .groupby(recent["report_date"].dt.strftime("%Y-%m-%d"))
            .size()
            .to_dict()
        )

    logging.debug(
        f"[api] Stats: {len(df)} total, {today_count} today, "
        f"{len(modality_counts)} modalities, {len(trend)} trend days."
    )

    return jsonify({
        "total_reports":    len(df),
        "today_reports":    today_count,
        "modality_counts":  modality_counts,
        "top_radiologists": rad_counts,
        "daily_trend":      trend,
    })


@api_bp.route("/exit", methods=["POST"])
@login_required
def exit_app():
    """
    Gracefully shuts down the RadiSearch server on user request.

    Called when the user clicks the "Exit App" button in any page's topbar.
    The browser sends a POST, we respond with a redirect to /closed, then
    schedule os._exit(0) on a short background timer so the HTTP response
    has time to reach the browser before the process dies.

    WHY os._exit() AND NOT sys.exit()?
    sys.exit() raises SystemExit which can be caught by threads and may not
    terminate cleanly. os._exit(0) unconditionally kills the process.

    WHY THE 0.5s DELAY?
    We need the 200 response + redirect to land in the browser before the
    server stops accepting connections. Without the delay the browser gets
    a connection-refused error instead of the /closed page.
    """
    user_email = session.get("user", {}).get("email", "unknown")
    logging.info(f"[api] Exit requested by user: {user_email}")

    def _shutdown():
        time.sleep(0.5)
        logging.info("[api] Shutting down via user request.")
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()

    # Redirect the browser to /closed before the process dies
    return jsonify({"redirect": "/closed"}), 200