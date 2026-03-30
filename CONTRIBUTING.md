# Contributing to RadiSearch

This document explains how to set up a development environment, understand the codebase, and make changes safely.

---

## Development Setup

```bash
# Clone and enter the project
git clone https://github.com/your-org/radisearch.git
cd radisearch

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env — at minimum set the PostgreSQL credentials and DEV_MODE=true

# Run
python app.py
```

With `DEV_MODE=true` you do not need Azure AD credentials. The app auto-logs you in as "Dev User".

---

## Project Conventions

### Where things go

| What you're adding | Where it goes |
|---|---|
| New configuration setting | `config.py` only — never `os.getenv()` elsewhere |
| New database query | `database.py` |
| New search logic | `search.py` |
| New Ollama / LLM helper | `llm.py` |
| New auth logic | `auth.py` |
| New page route (returns HTML) | `routes/pages.py` |
| New JSON API endpoint | `routes/api.py` |
| New search/filter endpoint | `routes/search_routes.py` |
| New LLM endpoint | `routes/llm_routes.py` |
| New HTML page | `templates/` |

### Imports between modules

The import dependency graph is strictly one-directional — no circular imports:

```
app.py
  └── routes/* (blueprints)
        └── search.py, llm.py, auth.py
              └── embeddings.py, database.py
                    └── config.py
```

`config.py` imports nothing from this project. `database.py` imports only `config`. And so on up the chain. Never import a higher-level module from a lower-level one.

### Adding a new route

1. Decide which Blueprint it belongs to (pages / api / search / llm)
2. Add the route function to the appropriate file in `routes/`
3. Protect it with `@login_required` unless it must be publicly accessible (like `/heartbeat` or `/closed`)
4. Call `reset_activity()` from `routes/api.py` at the start of any route that represents a real user action

Example:

```python
from flask import Blueprint, jsonify
from auth import login_required
from routes.api import reset_activity

my_bp = Blueprint("my", __name__)

@my_bp.route("/my_endpoint", methods=["POST"])
@login_required
def my_endpoint():
    reset_activity()
    # ... your logic
    return jsonify({"ok": True})
```

Then register it in `app.py`:
```python
from routes.my_module import my_bp
app.register_blueprint(my_bp)
```

### Adding a new config setting

Only ever add settings in `config.py`:

```python
# In config.py
MY_SETTING = os.getenv("MY_SETTING", "default_value")
```

Then import it wherever needed:
```python
from config import MY_SETTING
```

Add it to `.env.example` with a comment explaining what it does.

### Modifying HTML templates

Templates use Flask's Jinja2 engine. Variables passed from the route are available with `{{ variable_name }}`.

Both `search_mode.html` and `llm_mode.html` share the same CSS design tokens — if you change a colour or spacing value, change it in both files. The `:root { }` block at the top of each template defines all shared variables. Keep them in sync.

---

## Making Changes to the Embedding Pipeline

The embedding pipeline (`embeddings.py`) runs in a background thread. A few important rules:

- **Always use `_build_lock`** if you add any code that writes to `index` or `meta_data` — the lock prevents two threads rebuilding simultaneously
- **Never import `model`, `index`, or `meta_data` at module level** in `search.py` or other files — always import inside the function so you get the current value after initialisation
- **The `progress` dict is read by `/progress` every 500ms** — update it accurately during any rebuild so the loading bar works

---

## Changing the SQL Query

Edit `fetch_data.sql` directly. The query must return these columns (names must match exactly):

```
clean_report, patient_urn, visit_number, radiologist, modality, clinic, report_date
```

After changing the SQL, delete the cache files so they rebuild on next launch:
```bash
del faiss_index.bin embedding_meta.pkl   # Windows
rm faiss_index.bin embedding_meta.pkl    # Mac/Linux
```

---

## Environment Variables

Never hardcode credentials. Always add them to `.env` and read via `config.py`. The `config.validate()` function runs at startup and logs a warning for any missing required variable — check the terminal output when starting the app.

---

## Security Notes

- `DEV_MODE=true` **completely disables authentication**. It must never be set on a server accessible to other users.
- The `FLASK_SECRET_KEY` signs the session cookie. If it changes, all active user sessions are immediately invalidated (everyone gets logged out). Keep it stable in production.
- The `AZURE_ALLOWED_DOMAIN` check in `auth.py` is the critical access control gate. Test that it correctly rejects accounts from other domains before deploying.
- Never log the full session object, JWT tokens, or the client secret.
