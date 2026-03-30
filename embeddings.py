# =============================================================================
# embeddings.py — ChromaDB Vector Store with Bedrock Titan Embeddings support
# =============================================================================
#
# WHAT CHANGED FROM LOCAL VERSION:
#   Added Bedrock Titan Embeddings as an alternative to sentence-transformers.
#   When USE_BEDROCK_EMBED=true, report texts are embedded using Amazon Titan
#   Embeddings V2 via AWS Bedrock instead of the local model.
#
# EMBEDDING MODE SELECTION:
#   USE_BEDROCK_EMBED=false → sentence-transformers all-MiniLM-L6-v2 (local, free)
#   USE_BEDROCK_EMBED=true  → Amazon Titan Embeddings V2 (AWS, ~$0.02/1M tokens)
#
# NOTE: You cannot mix embedding modes in the same ChromaDB collection.
# If you switch USE_BEDROCK_EMBED, delete chroma_db/ to force a full rebuild.
#
# TITAN EMBEDDINGS V2:
#   - 1024-dimensional vectors (vs 384 for MiniLM)
#   - Better quality for clinical text
#   - Cost: ~$0.02 per 1M tokens
#   - Initial index of 100k reports: ~50M tokens ≈ $1.00 one-time cost
#   - Daily re-embedding of new reports only: ~$0.01/day
#
# PUBLIC API (unchanged):
#   model                      SentenceTransformer OR None (Bedrock mode)
#   chroma_collection          ChromaDB collection
#   meta_data                  list of all records
#   progress                   build progress dict
#   initialize_embeddings()    called at startup
#   background_refresh_watcher() daemon thread
# =============================================================================

import os
import time
import pickle
import logging
import threading
from datetime import datetime

import chromadb
from chromadb.config import Settings

from config import (
    MODEL_NAME,
    EMBEDDING_META,
    BATCH_SIZE,
    USE_PKL_DATA,
    CHROMA_DB_PATH,
    USE_BEDROCK_EMBED,   # controls embeddings only — USE_BEDROCK controls LLM only
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BEDROCK_EMBED_MODEL,
    USE_S3,
    AWS_S3_BUCKET,
    AWS_S3_PKL_KEY,
)

# =============================================================================
# MODULE-LEVEL STATE
# =============================================================================

# Local sentence-transformer model — loaded only when USE_BEDROCK_EMBED=false.
# When USE_BEDROCK=true (LLM on Bedrock) but USE_BEDROCK_EMBED=false,
# the local model still loads here — embeddings stay free and local.
model = None
if not USE_BEDROCK_EMBED:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    logging.info(f"[embeddings] Local model loaded: {MODEL_NAME}")
else:
    logging.info(f"[embeddings] Bedrock embed mode — using {BEDROCK_EMBED_MODEL}")

chroma_collection = None
meta_data: list | None = None
progress: dict = {"current": 0, "total": 0, "status": "loading"}
_build_lock = threading.Lock()

COLLECTION_NAME  = "radisearch_reports"
_LAST_BUILT_FILE = os.path.join("chroma_db", ".last_built")

# Track which embedding mode was used to build the current index
# so we can detect mode switches and force a rebuild
_EMBED_MODE_FILE = os.path.join("chroma_db", ".embed_mode")


# =============================================================================
# BEDROCK EMBEDDINGS
# =============================================================================

_bedrock_embed_client = None

def _get_bedrock_embed_client():
    """Returns boto3 Bedrock runtime client for embeddings."""
    global _bedrock_embed_client
    if _bedrock_embed_client is None:
        import boto3
        session = boto3.Session(
            aws_access_key_id     = AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
            region_name           = AWS_REGION,
        )
        _bedrock_embed_client = session.client("bedrock-runtime")
        logging.info(f"[embeddings] Bedrock embed client created ({BEDROCK_EMBED_MODEL})")
    return _bedrock_embed_client


def _embed_bedrock(texts: list[str]) -> list[list[float]]:
    """
    Embeds a batch of texts using Amazon Titan Embeddings V2.

    Titan V2 produces 1024-dimensional vectors — richer than MiniLM's 384.
    Processes one text at a time (Titan does not support batch embedding).

    Cost: ~$0.02 per 1M tokens. A typical report is ~150 tokens.
    100k reports ≈ 15M tokens ≈ $0.30 one-time indexing cost.
    """
    import json
    client = _get_bedrock_embed_client()
    embeddings = []

    for text in texts:
        # Titan V2 supports up to 8192 tokens — truncate if needed
        truncated = text[:6000] if len(text) > 6000 else text

        response = client.invoke_model(
            modelId     = BEDROCK_EMBED_MODEL,
            body        = json.dumps({
                "inputText":  truncated,
                "dimensions": 1024,   # Titan V2 supports 256, 512, or 1024
                "normalize":  True,   # L2-normalised for cosine similarity
            }),
            contentType = "application/json",
            accept      = "application/json",
        )
        result = json.loads(response["body"].read())
        embeddings.append(result["embedding"])

    return embeddings


def _embed_local(texts: list[str]) -> list[list[float]]:
    """Embeds texts using local sentence-transformers."""
    return model.encode(texts, show_progress_bar=False).tolist()


def _embed(texts: list[str]) -> list[list[float]]:
    """Routes embedding to Bedrock or local model based on USE_BEDROCK_EMBED flag."""
    if USE_BEDROCK_EMBED:
        return _embed_bedrock(texts)
    else:
        return _embed_local(texts)


def _current_embed_mode() -> str:
    """Returns a string identifying current embedding mode for cache validation."""
    return f"bedrock:{BEDROCK_EMBED_MODEL}" if USE_BEDROCK_EMBED else f"local:{MODEL_NAME}"


# =============================================================================
# S3 DATA SOURCE
# =============================================================================

def _load_pkl_from_s3() -> list:
    """
    Downloads embedding_meta.pkl from S3 and returns the records.
    Used when USE_S3=true and running on EC2.
    """
    import boto3, io
    logging.info(f"[embeddings] Downloading pkl from s3://{AWS_S3_BUCKET}/{AWS_S3_PKL_KEY}")
    s3 = boto3.client(
        "s3",
        aws_access_key_id     = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
        region_name           = AWS_REGION,
    )
    obj    = s3.get_object(Bucket=AWS_S3_BUCKET, Key=AWS_S3_PKL_KEY)
    data   = obj["Body"].read()
    records = pickle.loads(data)
    logging.info(f"[embeddings] Downloaded {len(records):,} records from S3.")
    return records


def _load_pkl() -> list:
    """
    Loads records from pkl file — either local filesystem or S3.
    USE_S3=true  → downloads from S3 bucket
    USE_S3=false → reads from local embedding_meta.pkl
    """
    if USE_S3:
        return _load_pkl_from_s3()
    else:
        if not os.path.exists(EMBEDDING_META):
            raise FileNotFoundError(
                f"USE_PKL_DATA=true but '{EMBEDDING_META}' not found. "
                "Place embedding_meta.pkl in the project folder and restart."
            )
        with open(EMBEDDING_META, "rb") as f:
            return pickle.load(f)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _get_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def _existing_count(client) -> int:
    try:
        return client.get_collection(COLLECTION_NAME).count()
    except Exception:
        return 0


def _embed_mode_changed() -> bool:
    """
    Returns True if the embedding mode has changed since the last build.
    Detects when USE_BEDROCK_EMBED is flipped, requiring a full rebuild.
    """
    if not os.path.exists(_EMBED_MODE_FILE):
        return False
    try:
        with open(_EMBED_MODE_FILE) as f:
            saved = f.read().strip()
        return saved != _current_embed_mode()
    except Exception:
        return False


def _save_embed_mode():
    """Records current embedding mode to disk."""
    os.makedirs(os.path.dirname(_EMBED_MODE_FILE), exist_ok=True)
    with open(_EMBED_MODE_FILE, "w") as f:
        f.write(_current_embed_mode())


def _mark_done() -> None:
    from routes.api import set_embeddings_ready
    progress["status"] = "done"
    set_embeddings_ready()


def _record_to_metadata(r: dict) -> dict:
    report_date = r.get("report_date", "")
    visit_dt    = r.get("visit_datetime", "")
    if hasattr(report_date, "strftime"):
        report_date = report_date.strftime("%Y-%m-%d")
    if hasattr(visit_dt, "strftime"):
        visit_dt = visit_dt.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "visit_number":     str(r.get("visit_number",    "")),
        "patient_urn":      str(r.get("patient_urn",     "")),
        "visit_datetime":   str(visit_dt),
        "clinic":           str(r.get("clinic",          "")),
        "radiologist":      str(r.get("radiologist",     "")),
        "modality":         str(r.get("modality",        "")),
        "exam_code":        str(r.get("exam_code",       "")),
        "exam_description": str(r.get("exam_description","")),
        "report_date":      str(report_date),
    }


# =============================================================================
# BUILD
# =============================================================================

def build_chroma(records: list) -> None:
    """
    Encodes all records and stores them in ChromaDB.
    Uses Bedrock Titan or local sentence-transformers based on USE_BEDROCK_EMBED.

    Bedrock note: Titan does not support batch embedding — each text is
    sent individually. This is slower but still fast enough at BATCH_SIZE=32.
    Progress bar updates reflect actual API calls made.
    """
    global chroma_collection, meta_data, progress

    with _build_lock:
        total = len(records)
        progress.update({"total": total, "current": 0, "status": "loading"})

        embed_mode = "Bedrock Titan" if USE_BEDROCK_EMBED else "local MiniLM"
        logging.info(f"[embeddings] Building ChromaDB index: {total:,} records using {embed_mode}...")

        client = _get_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Use smaller batches for Bedrock (API rate limits)
        batch_size = 16 if USE_BEDROCK_EMBED else BATCH_SIZE

        for start in range(0, total, batch_size):
            batch  = records[start : start + batch_size]
            texts  = [r.get("clean_report", "") or "" for r in batch]

            embeddings = _embed(texts)

            collection.upsert(
                ids        = [str(r.get("visit_number", f"rec_{start+i}"))
                               for i, r in enumerate(batch)],
                embeddings = embeddings,
                documents  = texts,
                metadatas  = [_record_to_metadata(r) for r in batch],
            )

            progress["current"] = min(start + batch_size, total)

            # Log every 10% for long Bedrock builds
            if progress["current"] % max(1, total // 10) < batch_size:
                pct = progress["current"] / total * 100
                logging.info(f"[embeddings] Build progress: {pct:.0f}% ({progress['current']:,}/{total:,})")

        chroma_collection = collection
        meta_data         = records
        _save_embed_mode()
        _mark_done()
        logging.info(f"[embeddings] Build complete — {total:,} records indexed using {embed_mode}.")


# =============================================================================
# INITIALIZE
# =============================================================================

def initialize_embeddings() -> None:
    """
    Called once at startup. Loads from ChromaDB if valid, builds if not.

    Extra check: if USE_BEDROCK_EMBED changed since the last build (embedding
    dimensions differ between Titan 1024 and MiniLM 384), forces a rebuild.
    """
    global chroma_collection, meta_data, progress

    logging.info(f"[embeddings] Connecting to ChromaDB at '{CHROMA_DB_PATH}'...")
    logging.info(f"[embeddings] Embedding mode: {_current_embed_mode()}")

    client   = _get_client()
    existing = _existing_count(client)
    logging.info(f"[embeddings] ChromaDB has {existing:,} existing records.")

    # Detect if embedding mode changed — different dimensions = must rebuild
    if existing > 0 and _embed_mode_changed():
        logging.warning(
            "[embeddings] Embedding mode changed since last build — "
            "rebuilding index with new embedding model."
        )
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        existing = 0

    if USE_PKL_DATA:
        # ── PKL / S3 MODE ─────────────────────────────────────────────────────
        try:
            records = _load_pkl()
        except FileNotFoundError as e:
            logging.error(f"[embeddings] {e}")
            _mark_done()
            return

        expected = len(records)
        logging.info(f"[embeddings] Data source has {expected:,} records.")

        if existing == expected:
            logging.info(
                f"[embeddings] ChromaDB matches data source ({existing:,}) "
                "— loading from disk."
            )
            chroma_collection = client.get_collection(COLLECTION_NAME)
            meta_data = records
            progress.update({"total": expected, "current": expected})
            _mark_done()
            return

        logging.info(f"[embeddings] Count mismatch — rebuilding index.")
        build_chroma(records)

    else:
        # ── DATABASE MODE ─────────────────────────────────────────────────────
        from database import fetch_data
        import pandas as pd
        df      = fetch_data()
        records = df.to_dict("records")
        expected = len(records)
        logging.info(f"[embeddings] PostgreSQL returned {expected:,} records.")

        if existing == expected:
            last_built = ""
            if os.path.exists(_LAST_BUILT_FILE):
                with open(_LAST_BUILT_FILE) as f:
                    last_built = f.read().strip()
            if last_built == datetime.today().strftime("%Y-%m-%d"):
                logging.info(f"[embeddings] ChromaDB up to date — loading.")
                chroma_collection = client.get_collection(COLLECTION_NAME)
                meta_data = records
                progress.update({"total": expected, "current": expected})
                _mark_done()
                return

        build_chroma(records)
        os.makedirs(os.path.dirname(_LAST_BUILT_FILE), exist_ok=True)
        with open(_LAST_BUILT_FILE, "w") as f:
            f.write(datetime.today().strftime("%Y-%m-%d"))


# =============================================================================
# BACKGROUND NOON REFRESH
# =============================================================================

def background_refresh_watcher() -> None:
    while True:
        time.sleep(3600)
        now = datetime.now()
        if now.hour == 12 and now.minute < 5:
            if USE_PKL_DATA:
                logging.info("[embeddings] Noon refresh skipped — PKL mode.")
            else:
                logging.info("[embeddings] Noon refresh triggered...")
                try:
                    from database import fetch_data
                    records = fetch_data().to_dict("records")
                    build_chroma(records)
                    logging.info("[embeddings] Noon refresh complete.")
                except Exception as e:
                    logging.error(f"[embeddings] Noon refresh failed: {e}")