# =============================================================================
# routes/pages.py — Full Page Routes
# =============================================================================
#
# WHAT THIS MODULE DOES:
#   Serves the three main HTML pages of the application, plus the /closed
#   page shown when the server shuts down due to inactivity.
#
#   Route            Template            Description
#   ─────────────────────────────────────────────────────────────────────
#   GET /            landing.html        Mode selector (LLM vs Custom Search)
#   GET /search_mode search_mode.html    Two-panel search + filter UI
#   GET /llm_mode    llm_mode.html       Chat-style LLM interface
#   GET /closed      (inline HTML)       Shown when server shuts down
#
# All routes except /closed are protected with @login_required.
# Each protected route passes the logged-in user's name and email
# to the template so the topbar can show the user name and Logout button.
#
# BLUEPRINT:
#   pages_bp — registered in app.py with app.register_blueprint(pages_bp)
# =============================================================================

from flask import Blueprint, render_template, session, request

from auth import login_required
from config import INACTIVITY_LIMIT, USE_BEDROCK, USE_BEDROCK_EMBED, USE_PKL_DATA, USE_S3
from llm import ollama_available

pages_bp = Blueprint("pages", __name__)


# =============================================================================
# HELPER
# =============================================================================

def _user() -> dict:
    """
    Returns the current user dict from the Flask session.
    Shorthand used by every protected route to pass user info to templates.

    Returns:
        {"name": "Dr Jane Smith", "email": "jane.smith@hospital.com"}
    """
    return session.get("user", {"name": "", "email": ""})


def _footer_context() -> str:
    """
    Builds the footer tech-stack label dynamically from .env flags.
    Reflects exactly what is active so the footer is always accurate.
    Examples:
      "local embeddings · AWS Bedrock (Claude 3 Haiku) · PostgreSQL"
      "local embeddings · Ollama · local pkl file — no data leaves your machine"
    """
    from config import BEDROCK_LLM_MODEL

    # Embeddings component
    embed = "Bedrock embeddings (Titan V2)" if USE_BEDROCK_EMBED else "local embeddings"

    # LLM component
    if USE_BEDROCK:
        model_short = BEDROCK_LLM_MODEL.split(".")[1] if "." in BEDROCK_LLM_MODEL else BEDROCK_LLM_MODEL
        llm = f"AWS Bedrock ({model_short})"
    else:
        llm = "Ollama"

    # Data source component
    if USE_S3:
        data_src = "AWS S3"
    elif USE_PKL_DATA:
        data_src = "local pkl file"
    else:
        data_src = "PostgreSQL"

    # Privacy note — only show when everything is local
    all_local = not USE_BEDROCK and not USE_BEDROCK_EMBED and not USE_S3 and USE_PKL_DATA
    suffix = " — no data leaves your machine" if all_local else ""

    return f"Powered by {embed} · {llm} · {data_src}{suffix}"


# =============================================================================
# ROUTES
# =============================================================================

@pages_bp.route("/")
@login_required
def home():
    """
    Landing page — the first page users see after logging in.
    Displays two mode-selection cards: LLM Search and Custom Search.
    """
    user = _user()
    return render_template(
        "landing.html",
        inactivity_limit = INACTIVITY_LIMIT,
        user_name        = user["name"],
        user_email       = user["email"],
        footer_label     = _footer_context(),
    )


@pages_bp.route("/search_mode")
@login_required
def search_mode():
    """
    Custom Search page — two-panel layout with:
      Left panel:  Semantic keyword search
      Right panel: Radiologist Report Metrics (filter by radiologist/modality/date)
      Tab:         Dashboard with report statistics and charts
    """
    user = _user()
    return render_template(
        "search_mode.html",
        inactivity_limit = INACTIVITY_LIMIT,
        user_name        = user["name"],
        user_email       = user["email"],
        footer_label     = _footer_context(),
    )


@pages_bp.route("/llm_mode")
@login_required
def llm_mode():
    """
    LLM chat page — natural language interface backed by a locally-running
    Ollama model. Checks Ollama availability at render time so the template
    can show the correct status indicator and offline banner immediately.
    """
    user       = _user()
    ok, models = ollama_available()

    return render_template(
        "llm_mode.html",
        inactivity_limit = INACTIVITY_LIMIT,
        user_name        = user["name"],
        user_email       = user["email"],
        ollama_ok        = ok,
        ollama_models    = models,
        footer_label     = _footer_context(),
    )


@pages_bp.route("/closed")
def closed():
    """
    Shown when the server has shut down.
    Accepts an optional ?reason= query parameter:
      - reason=user     → "You closed the application"
      - reason=inactivity (default) → "Shut down after 5 minutes of inactivity"
    No login_required — must be accessible after session expires.
    """
    reason = request.args.get("reason", "inactivity")

    if reason == "user":
        heading = "RadiSearch has been closed"
        message = "You chose to close the application.<br>Run <code>python app.py</code> to start it again."
    else:
        heading = "RadiSearch has closed"
        message = "The application shut down after 5 minutes of inactivity.<br>Run <code>python app.py</code> to start it again."

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RadiSearch — Closed</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'DM Sans',sans-serif;background:#0d1117;color:#e6edf3;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    height:100vh;gap:16px;text-align:center;padding:24px;}}
  h2{{font-size:22px;font-weight:600;}}
  p{{font-size:14px;color:#8b949e;max-width:360px;line-height:1.7;}}
  code{{background:#21262d;padding:3px 8px;border-radius:5px;font-size:13px;}}
  .footer{{margin-top:16px;font-size:12px;color:#4b5563;}}
</style>
</head>
<body>
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#8b949e"
       stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0 0h18"/>
  </svg>
  <h2>{heading}</h2>
  <p>{message}</p>
  <p class="footer">Created by Shubham Sakhuja</p>
</body>
</html>"""