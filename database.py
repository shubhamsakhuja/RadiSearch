# =============================================================================
# database.py — PostgreSQL Connection and Data Fetching
# =============================================================================
#
# WHAT THIS MODULE DOES:
#   Handles everything related to the PostgreSQL database:
#     - Opening and closing connections
#     - Loading the SQL query from fetch_data.sql
#     - Executing the query and returning results as a pandas DataFrame
#
# DESIGN PRINCIPLE — "Dumb data layer":
#   This module does only one thing: get data out of the database and return
#   it in a clean, typed DataFrame. It does no searching, no filtering, no
#   embedding. That separation means you can swap the database or change the
#   SQL query without touching any search or embedding logic.
#
# PUBLIC API (what other modules import from here):
#   fetch_data()  — connects, runs SQL, returns a typed DataFrame
#
# DEPENDENCIES:
#   config.py     — database credentials
#   fetch_data.sql — the SQL query (in the working directory)
# =============================================================================

import os
import logging

import psycopg2
import psycopg2.extras
import pandas as pd

from config import (
    POSTGRES_HOST,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    SQL_FILE,
)


# =============================================================================
# CONNECTION
# =============================================================================

def connect_postgres() -> psycopg2.extensions.connection:
    """
    Opens and returns a connection to the PostgreSQL database.

    Uses credentials loaded from config.py (which reads them from .env).

    DictCursor makes every row behave like a dictionary, so you can write
    row["radiologist"] instead of row[3]. Much safer when column order changes.

    The caller is responsible for calling conn.close() when done.
    We deliberately don't use a context manager here because connections
    are used in pd.read_sql(), which needs the connection to stay open
    until the entire query result is fetched.

    Raises:
        psycopg2.OperationalError — if the database is unreachable,
        credentials are wrong, or the database name doesn't exist.
    """
    return psycopg2.connect(
        host=POSTGRES_HOST,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        cursor_factory=psycopg2.extras.DictCursor,
    )


# =============================================================================
# SQL QUERY LOADER
# =============================================================================

def load_query() -> str:
    """
    Reads and returns the SQL query from fetch_data.sql.

    WHY A SEPARATE .sql FILE?
    Keeping the query in fetch_data.sql (rather than as a Python string)
    means radiologists or analysts can edit the SQL without touching Python code.
    It also gets proper SQL syntax highlighting in any code editor.

    The file is expected to be in the same directory as app.py (the working
    directory when you run the app). If you move the project, make sure
    fetch_data.sql moves with it.

    Returns:
        The full SQL query as a string.

    Raises:
        FileNotFoundError — if fetch_data.sql doesn't exist in the
        working directory. Check that SQL_FILE in config.py is correct.
    """
    # __file__ is the path to this database.py file.
    # dirname gives us its containing folder.
    # We build the SQL path relative to that folder so it works regardless
    # of which directory the user runs the app from.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sql_path = os.path.join(base_dir, SQL_FILE)

    if not os.path.exists(sql_path):
        raise FileNotFoundError(
            f"[database] SQL file not found: {sql_path}\n"
            f"Make sure '{SQL_FILE}' exists in the same directory as app.py."
        )

    with open(sql_path, "r", encoding="utf-8") as f:
        query = f.read()

    logging.debug(f"[database] Loaded SQL query from {sql_path}")
    return query


# =============================================================================
# DATA FETCHER
# =============================================================================

def fetch_data() -> pd.DataFrame:
    """
    Connects to the database, runs the SQL query, and returns the results
    as a clean, typed pandas DataFrame ready for embedding.

    Post-processing applied:
      - patient_urn is cast to str  — prevents integer formatting issues
        (e.g. URN "00123456" becoming 123456 if stored as INT in the DB)
      - report_date is cast to datetime — enables date filtering and
        comparisons like df[df["report_date"] >= "2025-01-01"]

    Returns:
        pd.DataFrame with at minimum these columns (from fetch_data.sql):
          - clean_report  (str)       — the report text to be embedded
          - patient_urn   (str)       — patient identifier
          - visit_number  (str/int)   — hospital visit number
          - radiologist   (str)       — reporting radiologist's name
          - modality      (str)       — scan type (CT, MRI, X-Ray, etc.)
          - clinic        (str)       — clinic or department name
          - report_date   (datetime)  — date the report was created

    Raises:
        psycopg2.OperationalError — if the database connection fails.
        psycopg2.ProgrammingError — if the SQL query has a syntax error.
    """
    logging.info("[database] Connecting to PostgreSQL...")
    conn = connect_postgres()

    try:
        query = load_query()
        logging.info("[database] Executing SQL query...")

        # pd.read_sql runs the query and loads the entire result set into
        # a DataFrame in one call. This is efficient for moderate data sizes
        # (up to ~100k rows). For very large datasets, consider chunked reading.
        df = pd.read_sql(query, conn)

    finally:
        # Always close the connection, even if pd.read_sql raises an exception.
        # Leaving connections open exhausts the database's connection pool.
        conn.close()
        logging.debug("[database] Connection closed.")

    # ── Type casting ──────────────────────────────────────────────────────────

    # Cast patient_urn to string.
    # PostgreSQL may return it as an integer (bigint) even if it looks like
    # a padded number ("00123456"). Casting to str preserves leading zeros.
    if "patient_urn" in df.columns:
        df["patient_urn"] = df["patient_urn"].astype(str)

    # Parse report_date to proper Python datetime objects.
    # This allows pandas date comparisons:
    #   df[df["report_date"] >= pd.to_datetime("2025-03-01")]
    # Without this, dates would be plain strings and comparisons wouldn't work.
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"])

    logging.info(f"[database] Fetched {len(df)} reports from database.")
    return df