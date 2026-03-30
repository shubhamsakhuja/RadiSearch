# =============================================================================
# app.py — RadiSearch Application Entry Point
# =============================================================================
#
# WHAT THIS FILE DOES:
#   This is the only file you run directly:
#       python app.py
#
#   Its sole responsibilities are:
#     1. Configure the Flask app (secret key, session settings)
#     2. Register all Blueprints (auth, pages, api, search, llm)
#     3. Start background threads (inactivity watcher, noon refresh)
#     4. Build or load embeddings in a background thread
#     5. Open the browser and start the Flask development server
#
#   All application logic lives in the other modules.
#   This file intentionally contains NO route handlers, NO business logic,
#   and NO HTML — it is purely wiring and startup.
#
# HOW TO RUN:
#   cd radisearch/
#   python app.py
#
# REQUIREMENTS:
#   pip install -r requirements.txt
#
# PROJECT STRUCTURE:
#   app.py                  ← this file (entry point)
#   config.py               ← all settings and env variables
#   database.py             ← PostgreSQL connection and data fetching
#   embeddings.py           ← ChromaDB vector index build, load, cache
#   search.py               ← run_search(), highlight_text(), build_table_html()
#   llm.py                  ← Ollama helpers and intent parser
#   auth.py                 ← Azure AD SSO, login_required decorator
#   routes/
#     __init__.py           ← exports all blueprints
#     pages.py              ← /, /search_mode, /llm_mode, /closed
#     api.py                ← /heartbeat, /ping, /progress, /stats, etc.
#     search_routes.py      ← /search, /filter_reports
#     llm_routes.py         ← /llm_query, /clear_session
#   templates/
#     landing.html          ← mode selection page
#     search_mode.html      ← custom search + dashboard page
#     llm_mode.html         ← LLM chat page
#     auth_error.html       ← login error page
#     logged_out.html       ← logout confirmation page
# =============================================================================

import os
import sys
import time
import logging
import threading
import webbrowser

# ── Environment setup ─────────────────────────────────────────────────────────
# Force CPU-only mode for all ML libraries.
# Must be set BEFORE importing numpy or sentence_transformers.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # Hide all GPUs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Prevent crash on Windows/Mac
os.environ["OMP_NUM_THREADS"]      = "1"     # Limit CPU threads

# Suppress output if running without a terminal (e.g. as a Windows service)
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%H:%M:%S",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from flask import Flask

import config
from auth import auth_bp
from routes import pages_bp, api_bp, search_bp, llm_bp
from embeddings import initialize_embeddings, background_refresh_watcher
from routes.api import get_last_activity, reset_activity, is_embeddings_ready


# =============================================================================
# CREATE FLASK APP
# =============================================================================

app = Flask(__name__)

# Secret key — required for encrypting the session cookie.
# Set in .env as FLASK_SECRET_KEY. If missing, validate() will warn at startup.
app.secret_key = config.FLASK_SECRET_KEY

# Session cookie security settings
app.config["SESSION_COOKIE_HTTPONLY"] = True   # JS cannot read the cookie
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # Blocks cross-site request forgery


# =============================================================================
# REGISTER BLUEPRINTS
# =============================================================================
# Each Blueprint is a self-contained group of routes defined in its own module.
# Registering them here attaches all their routes to the Flask app.
# Order does not matter for functionality but auth_bp first is a clear convention.

app.register_blueprint(auth_bp)    # /auth/login, /auth/callback, /auth/logout
app.register_blueprint(pages_bp)   # /, /search_mode, /llm_mode, /closed
app.register_blueprint(api_bp)     # /heartbeat, /ping, /progress, /stats, etc.
app.register_blueprint(search_bp)  # /search, /filter_reports
app.register_blueprint(llm_bp)     # /llm_query, /clear_session


# =============================================================================
# INACTIVITY WATCHER
# =============================================================================

def inactivity_watcher() -> None:
    """
    Background daemon thread that checks every second whether the user
    has been inactive for longer than INACTIVITY_LIMIT (default: 5 minutes).

    When the limit is exceeded, os._exit(0) immediately terminates the
    entire process including all background threads. This is intentional —
    the app holds patient report data in RAM and should not linger.

    os._exit(0) is used instead of sys.exit() because sys.exit() raises
    SystemExit which can be caught by threads. os._exit() is unconditional.

    The inactivity timer is reset by:
      - Any /ping call (triggered by user clicking buttons in the UI)
      - Any /search or /filter_reports request
      - Any /llm_query request
    """
    while True:
        time.sleep(1)
        # Do not count down while embeddings are still being built.
        # For 100k records this can take 20-30 minutes on CPU — we must
        # not shut down the app before the data is even loaded.
        if not is_embeddings_ready():
            continue
        elapsed = time.time() - get_last_activity()
        if elapsed > config.INACTIVITY_LIMIT:
            logging.info(
                f"[app] Inactivity limit reached "
                f"({config.INACTIVITY_LIMIT}s) — shutting down."
            )
            os._exit(0)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # ── Validate configuration ────────────────────────────────────────────────
    # Logs warnings for any missing .env variables.
    # The app can still start with missing config (useful during development)
    # but will fail gracefully when the missing feature is actually used.
    config.validate()

    # ── Startup diagnostics ───────────────────────────────────────────────────
    print(f"  DEV_MODE         : {config.DEV_MODE}")
    print(f"  USE_PKL_DATA     : {config.USE_PKL_DATA}")
    print(f"  USE_BEDROCK      : {config.USE_BEDROCK}")
    print(f"  USE_BEDROCK_EMBED: {config.USE_BEDROCK_EMBED}")
    print(f"  USE_S3           : {config.USE_S3}")
    if config.USE_PKL_DATA and not config.USE_S3:
        _pkl = "embedding_meta.pkl"
        if os.path.exists(_pkl):
            import pickle as _pickle
            with open(_pkl, "rb") as _f:
                _tmp = _pickle.load(_f)
            print(f"  embedding_meta.pkl  : {len(_tmp):,} records")
            del _tmp
        else:
            print(f"  embedding_meta.pkl  : NOT FOUND — app will error on startup")
    if os.path.exists(config.CHROMA_DB_PATH):
        print(f"  chroma_db/          : exists (index loads from disk)")
    else:
        print(f"  chroma_db/          : not found (index will be built — first run)")

    # ── Start background threads ──────────────────────────────────────────────

    # Inactivity watcher — shuts down the app after 5 minutes of no activity.
    # daemon=True means it stops automatically when the main process exits.
    threading.Thread(
        target = inactivity_watcher,
        daemon = True,
        name   = "InactivityWatcher",
    ).start()
    logging.info("[app] Inactivity watcher started.")

    # Noon refresh watcher — silently rebuilds embeddings at 12:00 each day.
    # daemon=True so it stops with the main process.
    threading.Thread(
        target = background_refresh_watcher,
        daemon = True,
        name   = "NoonRefreshWatcher",
    ).start()
    logging.info("[app] Noon refresh watcher started.")

    # Embedding initialiser — loads today's cache or rebuilds from the database.
    # daemon=False means this thread runs to completion even if the user closes
    # the browser — we want the build to finish so tomorrow's launch is fast.
    # The loading overlay in the browser polls /progress until status="done".
    threading.Thread(
        target = initialize_embeddings,
        daemon = False,
        name   = "EmbeddingInit",
    ).start()
    logging.info("[app] Embedding initialisation started in background.")

    # ── Open browser ──────────────────────────────────────────────────────────
    # Small delay so the Flask server has time to start before the browser
    # tries to connect. Without this, the browser may show a "connection refused"
    # error on the very first load.
    def _open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=_open_browser, daemon=True).start()

    # ── Start Flask server ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RadiSearch is starting...")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    app.run(
        host        = "0.0.0.0",   # Listen on all network interfaces so other
                                    # machines on the intranet can connect
        port        = 5000,
        debug       = False,        # Never True in production — disables the
                                    # reloader which would break background threads
        use_reloader= False,        # Explicitly off — same reason as debug=False
    )