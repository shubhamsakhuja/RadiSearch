# =============================================================================
# config.py — RadiSearch Application Configuration
# =============================================================================
#
# This is the SINGLE SOURCE OF TRUTH for all application settings.
# Every other module imports from here — nothing is hardcoded elsewhere.
#
# HOW IT WORKS:
#   - load_dotenv() reads your .env file from the same directory
#   - os.getenv() reads each value from the environment
#   - All constants are defined here and imported wherever needed
#
# YOUR .env FILE SHOULD CONTAIN:
# ─────────────────────────────────────────────────────────────────────────────
#   # PostgreSQL database
#   POSTGRES_HOST=localhost
#   POSTGRES_DATABASE=hospital_db
#   POSTGRES_USERNAME=admin
#   POSTGRES_PASSWORD=secret
#
#   # Azure AD SSO
#   AZURE_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
#   AZURE_CLIENT_SECRET=your-secret-from-azure-portal
#   AZURE_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
#   AZURE_ALLOWED_DOMAIN=hospital.com
#
#   # Flask session encryption key
#   # Generate with: python -c "import secrets; print(secrets.token_hex(32))"
#   FLASK_SECRET_KEY=your-long-random-string-here
# ─────────────────────────────────────────────────────────────────────────────
# =============================================================================

import os
from dotenv import load_dotenv

# Load the .env file — must be called before any os.getenv() calls.
# Looks for .env in the current working directory (where you run app.py from).
load_dotenv()


# =============================================================================
# POSTGRESQL DATABASE
# =============================================================================

POSTGRES_HOST     = os.getenv("POSTGRES_HOST")
POSTGRES_DB       = os.getenv("POSTGRES_DATABASE")
POSTGRES_USER     = os.getenv("POSTGRES_USERNAME")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")


# =============================================================================
# FILE PATHS
# =============================================================================

# The SQL query file — edit this to change which reports are pulled
SQL_FILE = "fetch_data.sql"

# The pkl data source file (used when USE_PKL_DATA=true)
EMBEDDING_META = "embedding_meta.pkl"

# ChromaDB persistent storage folder.
# Created automatically on first run. Delete this folder to force a full rebuild.
# Default: "chroma_db/" in the project directory — portable, works on any machine.
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")


# =============================================================================
# EMBEDDING / SEARCH MODEL
# =============================================================================

# Sentence transformer model name.
# "all-MiniLM-L6-v2" converts text to 384-dimensional vectors.
# Downloaded automatically on first run (~90MB from HuggingFace).
MODEL_NAME = "all-MiniLM-L6-v2"

# Maximum number of search results returned per query.
TOP_N = 50

# Minimum cosine similarity score (0.0–1.0) for a result to be included.
# 0.40 = at least 40% similar to the query.
# Lower  → more results, less precise.
# Higher → fewer results, more precise.
SIMILARITY_THRESHOLD = 0.40

# Number of reports encoded per batch when building embeddings.
# Larger batches are faster but use more RAM. 64 is a safe default on CPU.
BATCH_SIZE = 64


# =============================================================================
# INACTIVITY TIMER
# =============================================================================

# Seconds of inactivity before the server shuts itself down.
# Default: 5 minutes (300 seconds).
# The timer is reset whenever the user performs any action (search, filter, etc.)
INACTIVITY_LIMIT = 15 * 60


# =============================================================================
# OLLAMA (LOCAL AI MODEL)
# =============================================================================

# URL of the locally-running Ollama server.
# Port 11434 is Ollama's default — only change if you've reconfigured Ollama.
OLLAMA_URL = "http://localhost:11434/api/chat"

# Preferred model name. The app auto-detects installed models and picks
# the best one from a preference list — this is only a fallback default.
# Recommended: "phi3" (fast on CPU) or "mistral" (higher quality, slower).
OLLAMA_MODEL = "mistral"

# How many seconds to wait for Ollama to respond before giving up.
# CPU inference can take 30–90 seconds, so we allow 2 full minutes.
OLLAMA_TIMEOUT = 120

# Preference order for model selection (most preferred first).
# The first installed model that matches a prefix in this list is used.
OLLAMA_MODEL_PREFERENCE = ["mistral", "phi3", "phi", "llama3", "llama2", "tinyllama", "gemma"]


# =============================================================================
# AZURE AD / MICROSOFT SSO
# =============================================================================
# Values come from your Azure Portal app registration.
# See auth.py for the full setup guide.

AZURE_CLIENT_ID      = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET  = os.getenv("AZURE_CLIENT_SECRET")
AZURE_TENANT_ID      = os.getenv("AZURE_TENANT_ID")

# Only emails ending in this domain are allowed to log in.
# e.g. "hospital.com" → only user@hospital.com accounts are permitted.
# Leave blank to allow any Microsoft account (not recommended).
AZURE_ALLOWED_DOMAIN = os.getenv("AZURE_ALLOWED_DOMAIN", "").lower()

# Microsoft OAuth2 authority URL for your tenant.
AZURE_AUTHORITY = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}"

# OAuth2 scopes — we only request User.Read (name + email).
# We never request access to mail, files, or anything else.
AZURE_SCOPE = ["User.Read"]


# =============================================================================
# FLASK
# =============================================================================

# Secret key used to sign and encrypt the browser session cookie.
# MUST be set in .env — the default below is only a development fallback.
# If this leaks, attackers could forge login sessions.
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-me-before-deploying")

# =============================================================================
# DEV MODE — bypasses Azure AD authentication for local testing
# =============================================================================
# Set DEV_MODE=true in your .env file to skip SSO entirely.
# Any visit to the app will be auto-logged in as a test user.
#
# WARNING: NEVER set DEV_MODE=true in production.
# It completely disables authentication — anyone can access patient data.
#
# Usage in .env:
#   DEV_MODE=true    ← bypasses Azure AD, auto-logs in as test user
#   DEV_MODE=false   ← normal Azure AD SSO (default)
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

# =============================================================================
# PKL DATA MODE — skip PostgreSQL, load data directly from embedding_meta.pkl
# =============================================================================
# Set USE_PKL_DATA=true in your .env to bypass the PostgreSQL connection
# and load report data directly from the embedding_meta.pkl file.
#
# USE THIS WHEN:
#   - Testing locally without a PostgreSQL connection
#   - Using the dummy dataset generated by generate_dummy.py
#   - The database is temporarily unavailable
#
# HOW IT WORKS:
#   - initialize_embeddings() checks this flag before connecting to the DB
#   - If true: loads records from EMBEDDING_META (.pkl) directly
#   - If false: connects to PostgreSQL via database.py (production default)
#   - The noon background refresh is also skipped in pkl mode
#     (the pkl file is static — nothing to refresh from)
#
# Usage in .env:
#   USE_PKL_DATA=true    ← use pkl file, skip PostgreSQL
#   USE_PKL_DATA=false   ← connect to PostgreSQL (default)
USE_PKL_DATA = os.getenv("USE_PKL_DATA", "false").lower() == "true"


# =============================================================================
# AWS CREDENTIALS
# =============================================================================

AWS_REGION            = os.getenv("AWS_REGION", "ap-southeast-2")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


# =============================================================================
# BEDROCK LLM  (replaces Ollama when USE_BEDROCK=true)
# =============================================================================

USE_BEDROCK        = os.getenv("USE_BEDROCK", "false").lower() == "true"
BEDROCK_LLM_MODEL  = os.getenv("BEDROCK_LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "4096"))


# =============================================================================
# BEDROCK EMBEDDINGS  (replaces sentence-transformers when USE_BEDROCK_EMBED=true)
# =============================================================================

USE_BEDROCK_EMBED   = os.getenv("USE_BEDROCK_EMBED", "false").lower() == "true"
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")


# =============================================================================
# S3 OPTIONAL DATA SOURCE
# =============================================================================

USE_S3         = os.getenv("USE_S3", "false").lower() == "true"
AWS_S3_BUCKET  = os.getenv("AWS_S3_BUCKET", "")
AWS_S3_PKL_KEY = os.getenv("AWS_S3_PKL_KEY", "embedding_meta.pkl")



# =============================================================================
# VALIDATION — warn at startup if critical values are missing
# =============================================================================
# Called by app.py at startup. Logs warnings for any missing required config
# so you get a clear error message rather than a cryptic crash later.

def validate():
    """
    Check that all required environment variables are set.
    Logs a WARNING for each missing value but does not crash.
    """
    import logging

    required = {
        "POSTGRES_HOST":       POSTGRES_HOST,
        "POSTGRES_DATABASE":   POSTGRES_DB,
        "POSTGRES_USERNAME":   POSTGRES_USER,
        "POSTGRES_PASSWORD":   POSTGRES_PASSWORD,
        "AZURE_CLIENT_ID":     AZURE_CLIENT_ID,
        "AZURE_CLIENT_SECRET": AZURE_CLIENT_SECRET,
        "AZURE_TENANT_ID":     AZURE_TENANT_ID,
        "AZURE_ALLOWED_DOMAIN": AZURE_ALLOWED_DOMAIN,
        "FLASK_SECRET_KEY": (
            FLASK_SECRET_KEY
            if FLASK_SECRET_KEY != "change-me-before-deploying"
            else None
        ),
    }

    if USE_BEDROCK or USE_BEDROCK_EMBED or USE_S3:
        required["AWS_ACCESS_KEY_ID"]     = AWS_ACCESS_KEY_ID
        required["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    if USE_S3:
        required["AWS_S3_BUCKET"] = AWS_S3_BUCKET or None

    missing = [k for k, v in required.items() if not v]
    if missing:
        for key in missing:
            logging.warning(f"[config] Missing required environment variable: {key}")
    else:
        logging.info("[config] All required environment variables are set.")

    logging.info("[config] Bedrock LLM  : " + (
        "ENABLED  — " + BEDROCK_LLM_MODEL + "  (" + AWS_REGION + ")"
        if USE_BEDROCK else "disabled — using local Ollama"
    ))
    logging.info("[config] Bedrock Embed: " + (
        "ENABLED  — " + BEDROCK_EMBED_MODEL
        if USE_BEDROCK_EMBED else "disabled — using local sentence-transformers"
    ))
    logging.info("[config] S3 data      : " + (
        "ENABLED  — s3://" + AWS_S3_BUCKET + "/" + AWS_S3_PKL_KEY
        if USE_S3 else "disabled — using local embedding_meta.pkl"
    ))

    return len(missing) == 0