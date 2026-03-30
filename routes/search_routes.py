# =============================================================================
# routes/search_routes.py — Custom Search and Filter Routes
# =============================================================================
#
# WHAT THIS MODULE DOES:
#   Handles the two POST endpoints that power the Custom Search page:
#
#   POST /search          — semantic search by clinical keyword
#   POST /filter_reports  — filter by radiologist / modality / clinic / date
#
#   Both routes receive JSON from the browser, call run_search() from
#   search.py, and return an HTML table string that the browser inserts
#   directly into the results div via innerHTML.
#
# WHY HTML, NOT JSON?
#   The results table is complex HTML (with badges, collapsible details,
#   export button, expand/collapse controls). Returning HTML directly from
#   the server means the browser just does one innerHTML assignment — no
#   client-side template rendering needed. This is simpler and faster for
#   a single-user intranet app where server-side rendering is fine.
#
# SEARCH HISTORY:
#   search_history is a module-level list of recent query strings.
#   api.py's /history route imports it from here to serve the history pills.
#   It is intentionally in-memory only — cleared on restart.
#
# BLUEPRINT:
#   search_bp — registered in app.py with app.register_blueprint(search_bp)
# =============================================================================

import logging

from flask import Blueprint, request

from auth import login_required
from search import run_search, build_table_html
from routes.api import reset_activity

search_bp = Blueprint("search", __name__)

# In-memory list of recent search queries — last 50 stored.
# Imported by routes/api.py to serve the /history endpoint.
# Cleared when the app restarts (intentional — no persistence needed).
search_history: list = []


# =============================================================================
# ROUTES
# =============================================================================

@search_bp.route("/search", methods=["POST"])
@login_required
def search():
    """
    Handles a semantic search request from the Custom Search panel.

    Request body (JSON):
        {"query": "pleural effusion"}

    Returns:
        HTML string — a complete results table inserted into the page
        by the browser with: document.getElementById('search-results').innerHTML = html

    Pipeline:
        1. Extract and validate the query string
        2. Add to search_history (for the history pills UI)
        3. Call run_search() with just the query — no filters
        4. Build and return the HTML results table

    The /search route does NOT accept modality/clinic/date filters.
    Those are only available in the Radiologist Report Metrics panel (/filter_reports).
    If the user wants a combined semantic + filter search, that is handled
    by the LLM mode via /llm_query which calls run_search() with both.

    On every search, reset_activity() is called so the inactivity timer
    restarts — the user is clearly active if they're searching.
    """
    reset_activity()

    data  = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        logging.debug("[search] Empty query received.")
        return '<p class="no-results">Enter a search query.</p>'

    # Add to history — avoid consecutive duplicates but allow repeats
    # after other queries (e.g. searching "cancer", then "fracture", then
    # "cancer" again should show "cancer" once at the top)
    if not search_history or search_history[-1] != query:
        search_history.append(query)
        # Keep history bounded — drop the oldest entry if over 50
        if len(search_history) > 50:
            search_history.pop(0)

    logging.info(f"[search] Query: '{query}'")

    results = run_search(search_query=query)

    logging.info(f"[search] Returned {len(results)} results for '{query}'")

    return build_table_html(results, query=query, result_id_prefix="s")


@search_bp.route("/filter_reports", methods=["POST"])
@login_required
def filter_reports():
    """
    Handles a filter request from the Radiologist Report Metrics panel.

    Request body (JSON) — all fields are optional, but at least one must
    be provided (enforced below to prevent accidentally dumping the entire
    database when the user clicks Filter without selecting anything):
        {
            "radiologist": "Dr Smith",      or null
            "modality":    "CT",            or null
            "clinic":      "City Clinic",   or null
            "start_date":  "2025-03-01",    or null
            "end_date":    "2025-03-31"     or null
        }

    Returns:
        HTML string — a complete results table, or a warning message if
        no filter criteria were provided.

    Guard — at least one filter required:
        If all five fields are null/empty, we return a warning rather than
        running the query. Without this guard, clicking Filter with nothing
        selected would return all 500+ records, which is slow and confusing.

    Pipeline:
        1. Extract filter parameters from JSON body
        2. Validate at least one filter is present
        3. Call run_search() with filter parameters only (no search_query)
        4. Build and return the HTML results table
    """
    reset_activity()

    data        = request.get_json(silent=True) or {}
    radiologist = data.get("radiologist") or None
    modality    = data.get("modality")    or None
    clinic      = data.get("clinic")      or None
    start_str   = data.get("start_date")  or None
    end_str     = data.get("end_date")    or None

    # Guard — refuse to return the entire dataset when no filters are set.
    # The user must select at least one of: radiologist, modality, clinic,
    # start_date, or end_date before the Filter button does anything useful.
    if not any([radiologist, modality, clinic, start_str, end_str]):
        logging.debug("[search] Filter called with no criteria — returning warning.")
        return (
            '<p class="no-results warn">'
            "Please select at least one filter (radiologist, modality, "
            "clinic, or date range) before filtering."
            "</p>"
        )

    logging.info(
        f"[search] Filter — radiologist={radiologist}, modality={modality}, "
        f"clinic={clinic}, start={start_str}, end={end_str}"
    )

    results = run_search(
        radiologist = radiologist,
        modality    = modality,
        clinic      = clinic,
        start_date  = start_str,
        end_date    = end_str,
    )

    logging.info(f"[search] Filter returned {len(results)} results.")

    # No query for highlighting — filter-only results have no text to mark
    return build_table_html(results, query=None, result_id_prefix="f")