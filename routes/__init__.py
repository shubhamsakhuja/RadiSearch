# =============================================================================
# routes/__init__.py — Routes Package Initialiser
# =============================================================================
#
# WHAT THIS FILE DOES:
#   Makes the `routes` directory a proper Python package so that app.py can
#   import from it cleanly:
#
#       from routes import pages_bp, api_bp, search_bp, llm_bp
#
#   Without this file, Python would not recognise `routes` as a package
#   and the imports above would fail with a ModuleNotFoundError.
#
# HOW FLASK BLUEPRINTS WORK:
#   Each file in routes/ defines one Blueprint — a self-contained group
#   of related URL routes. app.py imports all four Blueprints from here
#   and registers them on the Flask app with app.register_blueprint().
#
#   Blueprint      File                  Routes it handles
#   ─────────────────────────────────────────────────────────────────
#   pages_bp       pages.py              /  /search_mode  /llm_mode  /closed
#   api_bp         api.py                /heartbeat  /ping  /filter_options
#                                        /history  /ollama_status  /stats
#   search_bp      search_routes.py      /search  /filter_reports
#   llm_bp         llm_routes.py         /llm_query  /clear_session
#
# WHY SPLIT INTO FOUR BLUEPRINTS?
#   Each Blueprint groups routes by what they do, not just by URL pattern:
#   - pages_bp    → serves full HTML pages (returns render_template)
#   - api_bp      → returns JSON data (jsonify) — utility endpoints
#   - search_bp   → handles search and filter POST requests
#   - llm_bp      → handles streaming LLM chat requests
#
#   This means if you want to change how search works, you only open
#   search_routes.py. If you want to add a new dashboard stat, you only
#   open api.py. No hunting through a 3000-line file.
# =============================================================================

from .pages         import pages_bp
from .api           import api_bp
from .search_routes import search_bp
from .llm_routes    import llm_bp

# Expose all four blueprints as the public API of this package.
# app.py imports them with: from routes import pages_bp, api_bp, search_bp, llm_bp
__all__ = ["pages_bp", "api_bp", "search_bp", "llm_bp"]