# =============================================================================
# search.py — Core Search Logic (ChromaDB + in-memory meta_data)
# =============================================================================
#
# FIELDS USED:
#   visit_number, patient_urn, visit_datetime, report_date,
#   radiologist, modality, clinic, exam_code, exam_description, clean_report
#
# TWO SEARCH PATHS:
#   1. Semantic search  — ChromaDB vector similarity on clean_report text
#   2. Filter-only      — read from meta_data directly (instant, no ChromaDB)
#
# top_n=None means NO LIMIT — return everything that matches.
# =============================================================================

import re
import logging

import pandas as pd

from config import TOP_N, SIMILARITY_THRESHOLD


# =============================================================================
# HIGHLIGHT TEXT
# =============================================================================

def highlight_text(text: str, query: str) -> str:
    if not query or not text:
        return text
    stopwords = {
        "the","and","or","in","of","to","a","an","is","are","was","were",
        "for","with","on","at","by","from","as","be","been","has","have",
        "had","this","that","these","those","which","who","what","when",
        "where","how","all","any","not","but","can","will","would","should",
        "may","might","its","it","he","she","they","we","you","i","my","our",
    }
    words = [w for w in re.findall(r'\w+', query.lower()) if w not in stopwords]
    if not words:
        return text
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b',
        re.IGNORECASE,
    )
    return pattern.sub(r'<mark>\1</mark>', text)


# =============================================================================
# FILTER OPTIONS
# =============================================================================

def get_filter_options() -> dict:
    from embeddings import meta_data
    if not meta_data:
        return {"radiologists": [], "modalities": [], "clinics": []}
    return {
        "radiologists": sorted({str(r.get("radiologist","")) for r in meta_data if r.get("radiologist")}),
        "modalities":   sorted({str(r.get("modality",""))    for r in meta_data if r.get("modality")}),
        "clinics":      sorted({str(r.get("clinic",""))      for r in meta_data if r.get("clinic")}),
    }


# =============================================================================
# CHROMADB WHERE CLAUSE
# =============================================================================

def _build_where(
    radiologist:  str | None,
    modality:     str | None,
    clinic:       str | None,
    start_date:   str | None,
    end_date:     str | None,
    exam_code:    str | None = None,
) -> dict | None:
    conditions = []
    if radiologist:
        conditions.append({"radiologist": {"$eq": radiologist}})
    if modality:
        conditions.append({"modality":    {"$eq": modality}})
    if clinic:
        conditions.append({"clinic":      {"$eq": clinic}})
    if start_date:
        conditions.append({"report_date": {"$gte": str(start_date)}})
    if end_date:
        conditions.append({"report_date": {"$lte": str(end_date)}})
    if exam_code:
        conditions.append({"exam_code":   {"$eq": exam_code}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# =============================================================================
# CHROMA RECORD CONVERTER
# =============================================================================

def _chroma_to_record(meta: dict, doc: str, distance: float | None = None) -> dict:
    rec = dict(meta)
    rec["clean_report"] = doc
    for col in ("report_date", "visit_datetime"):
        try:
            rec[col] = pd.to_datetime(rec.get(col, ""))
        except Exception:
            rec[col] = None
    rec["_score"] = round(1.0 - distance, 4) if distance is not None else None
    return rec


# =============================================================================
# FILTER RECORDS IN-MEMORY
# =============================================================================

def _filter_meta(
    records:      list,
    radiologist:  str | None = None,
    modality:     str | None = None,
    clinic:       str | None = None,
    start_date:   str | None = None,
    end_date:     str | None = None,
    exam_code:    str | None = None,
    start_dt:     str | None = None,
    end_dt:       str | None = None,
    exam_desc:    str | None = None,
) -> list:
    """
    Apply all filters directly on the in-memory record list.
    Handles every filter field including exam_code, exam_description,
    and datetime range (for morning/afternoon/shift queries).
    """
    out = records

    if radiologist:
        out = [r for r in out if str(r.get("radiologist","")).upper() == radiologist.upper()]
    if modality:
        out = [r for r in out if str(r.get("modality","")).upper() == modality.upper()]
    if clinic:
        out = [r for r in out if str(r.get("clinic","")).upper() == clinic.upper()]
    if exam_code:
        out = [r for r in out if str(r.get("exam_code","")).upper() == exam_code.upper()]
    if exam_desc:
        ed = exam_desc.lower()
        out = [r for r in out if ed in str(r.get("exam_description","")).lower()]

    # Date range on report_date
    if start_date or end_date:
        sd = pd.to_datetime(start_date) if start_date else None
        ed = pd.to_datetime(end_date)   if end_date   else None
        filtered = []
        for r in out:
            d = r.get("report_date")
            if d is None:
                continue
            try:
                dt = pd.to_datetime(d)
                if sd and dt < sd:
                    continue
                if ed and dt > ed:
                    continue
                filtered.append(r)
            except Exception:
                pass
        out = filtered

    # Datetime range on visit_datetime (for time-of-day filters)
    if start_dt or end_dt:
        sdt = pd.to_datetime(start_dt) if start_dt else None
        edt = pd.to_datetime(end_dt)   if end_dt   else None
        filtered = []
        for r in out:
            vd = r.get("visit_datetime")
            if vd is None:
                continue
            try:
                vdt = pd.to_datetime(vd)
                if sdt and vdt < sdt:
                    continue
                if edt and vdt > edt:
                    continue
                filtered.append(r)
            except Exception:
                pass
        out = filtered

    return out


# =============================================================================
# RUN SEARCH — main entry point
# =============================================================================

def run_search(
    search_query:     str | None = None,
    radiologist:      str | None = None,
    modality:         str | None = None,
    clinic:           str | None = None,
    start_date:       str | None = None,
    end_date:         str | None = None,
    top_n:            int | None = TOP_N,
    exam_code:        str | None = None,
    exam_description: str | None = None,
    start_datetime:   str | None = None,
    end_datetime:     str | None = None,
) -> list:
    """
    Main search entry point. Returns list of record dicts.

    top_n=None means no limit — return all matching records.

    Paths:
      1. No search_query, no filters → return all from meta_data (instant)
      2. No search_query, filters    → filter meta_data in memory
      3. search_query                → ChromaDB semantic search + apply filters
    """
    from embeddings import chroma_collection, meta_data as emb_meta, _embed

    has_filters = any([
        radiologist, modality, clinic, start_date, end_date,
        exam_code, exam_description, start_datetime, end_datetime,
    ])
    has_query = bool(search_query and search_query.strip())

    # ── PATH 1 & 2: No semantic search needed ─────────────────────────────────
    if not has_query:
        if emb_meta is None:
            logging.warning("[search] meta_data not ready")
            return []

        if not has_filters:
            # Return everything — sorted by report_date descending
            logging.info(f"[search] No-filter path — {len(emb_meta):,} records")
            records = sorted(
                emb_meta,
                key=lambda r: r.get("report_date") or pd.Timestamp.min,
                reverse=True,
            )
        else:
            records = _filter_meta(
                emb_meta,
                radiologist  = radiologist,
                modality     = modality,
                clinic       = clinic,
                start_date   = start_date,
                end_date     = end_date,
                exam_code    = exam_code,
                exam_desc    = exam_description,
                start_dt     = start_datetime,
                end_dt       = end_datetime,
            )
            records.sort(
                key=lambda r: r.get("report_date") or pd.Timestamp.min,
                reverse=True,
            )
            logging.info(f"[search] Filter-only path → {len(records):,} records")

        # Ensure _score field exists
        for r in records:
            r.setdefault("_score", None)

        return records if top_n is None else records[:int(top_n)]

    # ── PATH 3: Semantic search via ChromaDB ──────────────────────────────────
    if chroma_collection is None:
        logging.warning("[search] ChromaDB not ready — falling back to meta_data text scan")
        return _text_scan_fallback(
            search_query, emb_meta,
            radiologist, modality, clinic, start_date, end_date,
            exam_code, exam_description, start_datetime, end_datetime,
            top_n,
        )

    try:
        query_vector = _embed([search_query.strip()])

        where = _build_where(radiologist, modality, clinic, start_date, end_date, exam_code)

        total     = chroma_collection.count()
        n_results = max(1, total if top_n is None else min(int(top_n), total))

        kwargs = {
            "query_embeddings": query_vector,
            "n_results":        n_results,
            "include":          ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = chroma_collection.query(**kwargs)

        records = []
        for meta, doc, dist in zip(
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - (dist / 2.0)
            if similarity >= SIMILARITY_THRESHOLD:
                records.append(_chroma_to_record(meta, doc, dist))

        # Apply extra filters not supported by ChromaDB where-clause
        if exam_description or start_datetime or end_datetime:
            records = _filter_meta(
                records,
                exam_desc = exam_description,
                start_dt  = start_datetime,
                end_dt    = end_datetime,
            )

        # Keyword fallback if semantic returns too few
        if len(records) < 5:
            fb = _keyword_fallback(
                search_query, where, top_n,
                existing_ids={r.get("visit_number") for r in records},
            )
            records = fb + records

        logging.info(f"[search] Semantic '{search_query}' → {len(records):,} results")
        return records if top_n is None else records[:int(top_n)]

    except Exception as e:
        logging.error(f"[search] Semantic search error: {e}")
        return []


# =============================================================================
# TEXT SCAN FALLBACK (when ChromaDB unavailable)
# =============================================================================

def _text_scan_fallback(
    query, records,
    radiologist, modality, clinic, start_date, end_date,
    exam_code, exam_description, start_datetime, end_datetime,
    top_n,
) -> list:
    if not records:
        return []
    filtered = _filter_meta(
        records,
        radiologist=radiologist, modality=modality, clinic=clinic,
        start_date=start_date, end_date=end_date, exam_code=exam_code,
        exam_desc=exam_description, start_dt=start_datetime, end_dt=end_datetime,
    )
    keywords = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
    matches = [
        r for r in filtered
        if any(kw in str(r.get("clean_report","")).lower() for kw in keywords)
    ]
    for r in matches:
        r.setdefault("_score", None)
    logging.info(f"[search] Text scan fallback → {len(matches):,} results")
    return matches if top_n is None else matches[:int(top_n)]


# =============================================================================
# KEYWORD FALLBACK
# =============================================================================

def _keyword_fallback(query, where, top_n, existing_ids) -> list:
    from embeddings import chroma_collection
    try:
        stopwords = {
            "the","and","or","in","of","to","a","an","is","are","was","were",
            "for","with","on","at","by","from","as","be","been","has","have",
        }
        keywords = [
            w.lower() for w in re.findall(r'\w+', query)
            if w.lower() not in stopwords and len(w) > 2
        ]
        if not keywords:
            return []

        kwargs = {"include": ["metadatas", "documents"]}
        if where:
            kwargs["where"] = where
        if top_n is not None:
            kwargs["limit"] = int(top_n) * 10

        results = chroma_collection.get(**kwargs)
        matches = []
        for meta, doc in zip(results["metadatas"], results["documents"]):
            if str(meta.get("visit_number")) in existing_ids:
                continue
            if any(kw in doc.lower() for kw in keywords):
                matches.append(_chroma_to_record(meta, doc))

        logging.info(f"[search] Keyword fallback → {len(matches)} extra matches")
        return matches if top_n is None else matches[:int(top_n)]
    except Exception as e:
        logging.error(f"[search] Keyword fallback error: {e}")
        return []


# =============================================================================
# BUILD TABLE HTML
# =============================================================================

def build_table_html(
    records:          list,
    query:            str | None = None,
    result_id_prefix: str        = "r",
) -> str:
    if not records:
        return '<div class="no-results">No reports found matching your criteria.</div>'

    count = len(records)
    rows  = ""

    for i, r in enumerate(records):
        report    = highlight_text(str(r.get("clean_report",""))[:1000], query or "")
        score     = r.get("_score")
        sim_cell  = f'<span class="sim-score">{score:.0%}</span>' if score is not None else "—"
        date_val  = r.get("report_date","")
        if hasattr(date_val, "strftime"):
            try:
                date_val = date_val.strftime("%d %b %Y")
            except Exception:
                pass
        exam_desc = str(r.get("exam_description",""))[:40] or "—"
        result_id = f"{result_id_prefix}_{i}"

        rows += f"""
        <tr>
            <td>{r.get('visit_number','')}</td>
            <td>{r.get('clinic','')}</td>
            <td><span class="badge">{r.get('modality','')}</span></td>
            <td>{r.get('radiologist','')}</td>
            <td>{date_val}</td>
            <td class="sim-cell">{sim_cell}</td>
            <td>
                <details id="{result_id}">
                    <summary>View Report</summary>
                    <div class="report-body">{report}</div>
                </details>
            </td>
        </tr>"""

    export_id = f"export_{result_id_prefix}"
    return f"""
    <div class="results-header">
        <span class="result-count">{count} report{'s' if count != 1 else ''} found</span>
        <div class="results-actions">
            <button class="btn btn-ghost btn-sm"
                    onclick="toggleAllDetails('{result_id_prefix}', true)">Expand All</button>
            <button class="btn btn-ghost btn-sm"
                    onclick="toggleAllDetails('{result_id_prefix}', false)">Collapse All</button>
            <button class="btn btn-ghost btn-sm" id="{export_id}"
                    onclick="exportCSV('{result_id_prefix}', {count})">Export CSV</button>
        </div>
    </div>
    <table class="results-table">
        <thead>
            <tr>
                <th>Visit</th><th>Clinic</th><th>Modality</th>
                <th>Radiologist</th><th>Date</th><th>Score</th><th>Report</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""