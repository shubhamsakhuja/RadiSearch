# =============================================================================
# fuzzy.py — Fuzzy Resolution Layer
# =============================================================================
#
# Sits between intent parsing and run_search().
# Takes the raw intent dict and resolves fuzzy/ambiguous values into exact
# database values before the search runs.
#
# Handles:
#   1. Radiologist name fuzzy matching (partial, reversed, typos)
#   2. Modality alias resolution (xray→CR, catscam→CT, etc.)
#   3. Clinic fuzzy matching (ER→EMERGENCY DEPT, ortho→ORTHOPAEDIC, etc.)
#   4. Exam description keyword resolution
#   5. Clinical synonym expansion (heart attack → cardiac/MI/infarction)
#   6. Relative date expressions (recently, lately, this morning, etc.)
#   7. Time-of-day expressions (morning shift, overnight, weekday, etc.)
#   8. Performance expressions (busiest, top performer, etc.)
#   9. Negation / exclusion flags
#   10. Multi-value OR conditions (CT or MRI, Smith or Jones)
#   11. Body region → search term mapping
#   12. Quantity/volume expressions (a few, tonnes of, etc.)
#   13. Australian seasonal dates (summer=Dec-Feb, FY=Jul-Jun)
# =============================================================================

import re
import logging
from datetime import datetime, timedelta, date
from difflib import get_close_matches


# =============================================================================
# MODALITY ALIASES
# =============================================================================

_MODALITY_ALIASES = {
    # CT
    "ct": "CT", "cat scan": "CT", "catscam": "CT", "catscan": "CT",
    "cat": "CT", "computed tomography": "CT", "ct scan": "CT",

    # MRI
    "mri": "MRI", "mri scan": "MRI", "magnetic resonance": "MRI",
    "magnetic": "MRI", "magnet": "MRI", "nmr": "MRI",

    # X-Ray — XR is the universal clinical code, CR/DX are what hospital systems store
    "xr": "CR", "x-ray": "CR", "xray": "CR", "x ray": "CR",
    "plain film": "CR", "plain xray": "CR", "radiograph": "CR", "chest film": "CR",
    "digital xray": "DX", "digital x-ray": "DX",

    # Ultrasound
    "us": "US", "ultrasound": "US", "ultra sound": "US",
    "sonogram": "US", "sonography": "US", "echo": "US",
    "echocardiogram": "US",

    # Nuclear Medicine
    "nm": "NM", "nuclear": "NM", "nuclear medicine": "NM",
    "nuc med": "NM", "isotope": "NM", "nuclear med": "NM",
    "scintigraphy": "NM", "bone scan": "NM",

    # Mammography
    "mg": "MG", "mammogram": "MG", "mammo": "MG",
    "mammography": "MG", "breast screen": "MG",

    # Angiography
    "xa": "XA", "angio": "XA", "angiogram": "XA",
    "angiography": "XA", "fluoroangio": "XA",

    # Fluoroscopy
    "rf": "RF", "fluoro": "RF", "fluoroscopy": "RF",
    "barium": "RF", "swallow": "RF",

    # PET
    "pt": "PT", "pet": "PT", "pet scan": "PT",
    "positron": "PT", "pet ct": "PT",

    # Other
    "ot": "OT", "other": "OT",
}


def resolve_modality(raw: str, available_modalities: list) -> str | None:
    """
    Resolves user's modality input to exact database value.

    Priority order:
      1. Exact match (case-insensitive) — always wins, no substitution
      2. Alias lookup for layman terms (xray, catscam, etc.)
         — only if the alias actually exists in the data
         — never silently substitute (e.g. XR→CR) if the user typed a real code
      3. Fuzzy match as last resort for typos

    IMPORTANT: If the user types an explicit modality code (e.g. "XR") that does
    NOT exist in the data, return None and let the search return no results.
    This is correct behaviour — the user asked for XR, not CR.
    """
    if not raw:
        return None

    upper   = raw.upper().strip()
    cleaned = raw.lower().strip()

    # 1. Exact match — highest priority
    if upper in available_modalities:
        return upper

    # 2. Alias lookup — covers both layman terms AND standard clinical codes
    #    that differ from what the hospital system uses internally.
    #    e.g. XR is the universal clinical code for X-Ray, but this hospital
    #    system stores it as CR (Computed Radiography). We map XR → CR.
    if cleaned in _MODALITY_ALIASES:
        alias = _MODALITY_ALIASES[cleaned]
        if alias in available_modalities:
            logging.info(f"[fuzzy] Modality alias: '{raw}' → '{alias}'")
            return alias
        # Try CR↔DX swap (both are X-Ray variants)
        if alias == "CR" and "DX" in available_modalities:
            return "DX"
        if alias == "DX" and "CR" in available_modalities:
            return "CR"

    # 3. Fuzzy match for typos (e.g. "CTT" → "CT", "MRi" → "MRI")
    close = get_close_matches(upper, available_modalities, n=1, cutoff=0.75)
    if close:
        logging.info(f"[fuzzy] Modality fuzzy: '{raw}' → '{close[0]}'")
        return close[0]

    return None


# =============================================================================
# CLINICAL SYNONYM EXPANSION
# =============================================================================

_CLINICAL_SYNONYMS = {
    # Cardiac
    "heart attack":      "myocardial infarction MI cardiac",
    "heart disease":     "cardiac coronary disease",
    "heart failure":     "cardiac failure CCF heart",
    "chest pain":        "chest pain cardiac angina",

    # Neurological
    "stroke":            "CVA cerebrovascular accident infarct",
    "brain bleed":       "haemorrhage hematoma hemorrhage intracranial",
    "seizure":           "epilepsy seizure fit convulsion",

    # Respiratory
    "breathing problem": "respiratory dyspnoea shortness of breath",
    "lung clot":         "pulmonary embolism PE thrombosis",
    "blood clot":        "DVT deep vein thrombosis PE embolism thrombus",
    "pneumonia":         "pneumonia consolidation infection",

    # Oncology
    "cancer":            "cancer malignancy carcinoma neoplasm tumour mass",
    "tumour":            "tumour tumor mass neoplasm malignancy",
    "lump":              "mass lesion nodule lump",

    # Musculoskeletal
    "broken bone":       "fracture break #",
    "broken leg":        "fracture femur tibia fibula lower limb",
    "broken arm":        "fracture humerus radius ulna upper limb",
    "arthritis":         "osteoarthritis OA degenerative joint disease",
    "back pain":         "lumbar spine vertebral disc herniation",
    "knee pain":         "knee joint patella meniscus",

    # Abdominal
    "belly pain":        "abdominal pain abdomen",
    "tummy pain":        "abdominal pain abdomen",
    "gallstones":        "cholelithiasis gallbladder calculi stones",
    "kidney stones":     "nephrolithiasis urolithiasis calculus kidney",
    "appendix":          "appendicitis appendix",

    # Vascular
    "varicose veins":    "varicose veins venous insufficiency",
    "blocked artery":    "stenosis occlusion arterial disease",

    # Body regions → search terms
    "head":    "brain skull head intracranial cranial",
    "chest":   "chest thorax lung cardiac pulmonary",
    "belly":   "abdomen abdominal bowel liver",
    "tummy":   "abdomen abdominal",
    "back":    "spine lumbar thoracic vertebral disc",
    "neck":    "cervical neck spine throat",
    "legs":    "lower limb femur tibia knee ankle",
    "arms":    "upper limb humerus radius elbow wrist",
    "bones":   "fracture skeletal bone cortex",
    "joints":  "joint arthritis MSK musculoskeletal",
    "heart":   "cardiac coronary heart",
    "lungs":   "lung pulmonary thorax respiratory",
    "brain":   "brain cerebral intracranial head",
    "kidneys": "renal kidney",
    "liver":   "hepatic liver",
}


def expand_clinical_query(query: str) -> str:
    """
    Expands layman/casual clinical terms to broader search terms.
    Returns original query if no expansion found.
    """
    if not query:
        return query
    lower = query.lower().strip()
    if lower in _CLINICAL_SYNONYMS:
        expanded = _CLINICAL_SYNONYMS[lower]
        logging.info(f"[fuzzy] Clinical expansion: '{query}' → '{expanded}'")
        return expanded
    # Check if any synonym key is a substring
    for key, expansion in _CLINICAL_SYNONYMS.items():
        if key in lower:
            expanded = re.sub(re.escape(key), expansion, lower, flags=re.IGNORECASE)
            logging.info(f"[fuzzy] Partial expansion: '{query}' → '{expanded}'")
            return expanded
    return query


# =============================================================================
# RADIOLOGIST NAME FUZZY MATCHING
# =============================================================================

def resolve_radiologist(raw: str, available: list) -> str | None:
    """
    Matches user's radiologist input to exact database value.

    Database format: "SURNAME, FIRSTNAME" (UPPER)
    User may say: "Dr Smith", "James Smith", "Smith", "smyth" (typo)
    """
    if not raw:
        return None

    # Strip common prefixes
    cleaned = re.sub(
        r'\b(dr|doctor|prof|professor|mr|mrs|ms|miss|a/prof)\b\.?\s*',
        '', raw, flags=re.IGNORECASE
    ).strip().upper()

    # Direct exact match
    if cleaned in available:
        return cleaned

    # Try "SURNAME, FIRSTNAME" format from "Firstname Surname" input
    parts = cleaned.split()
    if len(parts) == 2:
        reversed_name = f"{parts[1]}, {parts[0]}"
        if reversed_name in available:
            return reversed_name
        # Also try just the surname part
        surname_matches = [a for a in available if a.split(",")[0].strip() == parts[1]]
        if len(surname_matches) == 1:
            return surname_matches[0]
        if len(surname_matches) > 1:
            logging.info(f"[fuzzy] Multiple surname matches for '{raw}': {surname_matches}")
            # Return None — ambiguous, let AI handle it
            return None

    # Single word — match as surname
    if len(parts) == 1:
        surname_matches = [a for a in available if a.split(",")[0].strip() == cleaned]
        if len(surname_matches) == 1:
            return surname_matches[0]

    # Fuzzy match as last resort
    close = get_close_matches(cleaned, available, n=1, cutoff=0.7)
    if close:
        logging.info(f"[fuzzy] Radiologist fuzzy match: '{raw}' → '{close[0]}'")
        return close[0]

    # Try fuzzy match on just the surname part of available names
    surnames = {a.split(",")[0].strip(): a for a in available}
    close = get_close_matches(cleaned, list(surnames.keys()), n=1, cutoff=0.7)
    if close:
        matched = surnames[close[0]]
        logging.info(f"[fuzzy] Radiologist surname fuzzy: '{raw}' → '{matched}'")
        return matched

    return None


# =============================================================================
# CLINIC FUZZY MATCHING
# =============================================================================

_CLINIC_ALIASES = {
    "emergency":    ["EMERGENCY", "ED", "ER", "EMERGENCY DEPT", "A&E", "CASUALTY"],
    "icu":          ["ICU", "INTENSIVE CARE", "CRITICAL CARE"],
    "outpatient":   ["OUTPATIENT", "OPD", "OUT-PATIENT", "OUTPATIENT CENTRE"],
    "ortho":        ["ORTHOPAEDIC", "ORTHOPAEDICS", "ORTHOPEDIC", "ORTHOPEDICS"],
    "surgical":     ["SURGICAL", "SURGERY", "THEATRE"],
    "vascular":     ["VASCULAR"],
    "east":         ["EAST WING", "EAST"],
    "west":         ["WEST WING", "WEST"],
    "north":        ["NORTH CAMPUS", "NORTH"],
    "south":        ["SOUTH CAMPUS", "SOUTH"],
    "breast":       ["BREAST CLINIC", "BREAST SCREEN", "BREAST"],
    "cardiology":   ["CARDIOLOGY", "CARDIAC", "CORONARY"],
    "neurology":    ["NEUROLOGY", "NEURO"],
    "oncology":     ["ONCOLOGY", "CANCER", "HAEMATOLOGY"],
    "paediatric":   ["PAEDIATRIC", "PEDIATRIC", "PAEDS", "PEDS", "CHILDREN"],
    "renal":        ["RENAL", "NEPHROLOGY", "KIDNEY"],
    "respiratory":  ["RESPIRATORY", "PULMONARY", "CHEST CLINIC"],
}


def resolve_clinic(raw: str, available: list) -> str | None:
    """
    Resolves user's clinic input to exact database value.
    Handles abbreviations, partial names, and common aliases.
    """
    if not raw:
        return None
    cleaned = raw.upper().strip()

    # Exact match
    if cleaned in available:
        return cleaned

    # Check alias table
    lower = raw.lower().strip()
    for key, synonyms in _CLINIC_ALIASES.items():
        if lower == key or any(s.lower() == lower for s in synonyms):
            # Find matching available clinic
            for avail in available:
                if any(s.upper() in avail.upper() for s in synonyms):
                    return avail

    # Fuzzy match
    close = get_close_matches(cleaned, available, n=1, cutoff=0.6)
    if close:
        logging.info(f"[fuzzy] Clinic fuzzy: '{raw}' → '{close[0]}'")
        return close[0]

    # Substring match
    for avail in available:
        if cleaned in avail or avail in cleaned:
            return avail

    return None


# =============================================================================
# RELATIVE DATE RESOLUTION
# =============================================================================

def resolve_relative_dates(intent: dict) -> dict:
    """
    Resolves additional casual date expressions that the intent parser
    may not have caught. Operates on the full intent dict.

    Also handles Australian seasons (Southern Hemisphere) and financial year.
    """
    today    = datetime.today().date()
    modified = dict(intent)

    user_msg_lower = intent.get("_raw_user_msg", "").lower()

    # Already has dates from intent parser — don't override
    if modified.get("start_date") or modified.get("end_date"):
        return modified

    # Relative expressions → date ranges
    patterns = [
        # Recently / lately
        (r'\b(recently|lately|just recently)\b',
         lambda: (today - timedelta(days=7), today)),

        # Last few days / past few days
        (r'\b(last few days|past few days|past several days)\b',
         lambda: (today - timedelta(days=7), today)),

        # Last couple of weeks
        (r'\b(last couple of weeks|past couple of weeks|couple weeks)\b',
         lambda: (today - timedelta(days=14), today)),

        # Past month / last 30 days
        (r'\b(past month|last 30 days|last thirty days)\b',
         lambda: (today - timedelta(days=30), today)),

        # Beginning of year / early this year
        (r'\b(beginning of (the )?year|early this year|start of year)\b',
         lambda: (date(today.year, 1, 1), date(today.year, 3, 31))),

        # End of last year
        (r'\b(end of last year|late last year|end of previous year)\b',
         lambda: (date(today.year-1, 10, 1), date(today.year-1, 12, 31))),

        # First half / second half of year
        (r'\b(first half( of (the )?year)?|H1)\b',
         lambda: (date(today.year, 1, 1), date(today.year, 6, 30))),
        (r'\b(second half( of (the )?year)?|H2)\b',
         lambda: (date(today.year, 7, 1), date(today.year, 12, 31))),

        # Australian financial year (Jul 1 - Jun 30)
        (r'\b(financial year|FY|current FY|this FY)\b',
         lambda: _au_financial_year(today)),

        # Last financial year
        (r'\b(last financial year|last FY|previous FY|prior FY)\b',
         lambda: _au_last_financial_year(today)),

        # Australian seasons (Southern Hemisphere)
        (r'\b(this summer|last summer)\b',
         lambda: _au_season("summer", today)),
        (r'\b(this winter|last winter)\b',
         lambda: _au_season("winter", today)),
        (r'\b(this autumn|last autumn|this fall|last fall)\b',
         lambda: _au_season("autumn", today)),
        (r'\b(this spring|last spring)\b',
         lambda: _au_season("spring", today)),

        # Quarters
        (r'\b(Q1|first quarter)\b',
         lambda: (date(today.year, 1, 1), date(today.year, 3, 31))),
        (r'\b(Q2|second quarter)\b',
         lambda: (date(today.year, 4, 1), date(today.year, 6, 30))),
        (r'\b(Q3|third quarter)\b',
         lambda: (date(today.year, 7, 1), date(today.year, 9, 30))),
        (r'\b(Q4|fourth quarter|last quarter of year)\b',
         lambda: (date(today.year, 10, 1), date(today.year, 12, 31))),

        # Last quarter
        (r'\b(last quarter|previous quarter)\b',
         lambda: _last_quarter(today)),

        # Specific day names → last occurrence
        (r'\b(last monday|last tuesday|last wednesday|last thursday|last friday|last saturday|last sunday)\b',
         lambda m=None: _last_weekday(user_msg_lower, today)),

        # "Around [month]"
        (r'\baround (january|february|march|april|may|june|july|august|september|october|november|december)\b',
         lambda: _month_range(user_msg_lower, today)),

        # "In [month]" without year
        (r'\bin (january|february|march|april|may|june|july|august|september|october|november|december)\b',
         lambda: _month_range(user_msg_lower, today)),
    ]

    for pattern, date_fn in patterns:
        if re.search(pattern, user_msg_lower, re.IGNORECASE):
            try:
                result = date_fn()
                if result and len(result) == 2:
                    modified["start_date"] = result[0].strftime("%Y-%m-%d")
                    modified["end_date"]   = result[1].strftime("%Y-%m-%d")
                    logging.info(f"[fuzzy] Date resolved: {pattern} → {modified['start_date']} to {modified['end_date']}")
                    break
            except Exception as e:
                logging.warning(f"[fuzzy] Date resolution failed for '{pattern}': {e}")

    return modified


def _au_financial_year(today: date) -> tuple:
    """Australian FY: Jul 1 to Jun 30."""
    if today.month >= 7:
        return date(today.year, 7, 1), date(today.year + 1, 6, 30)
    return date(today.year - 1, 7, 1), date(today.year, 6, 30)

def _au_last_financial_year(today: date) -> tuple:
    start, end = _au_financial_year(today)
    return date(start.year - 1, 7, 1), date(start.year, 6, 30)

def _au_season(season: str, today: date) -> tuple:
    """Australian seasons (Southern Hemisphere)."""
    seasons = {
        "summer": [(12, 1), (2, 28)],   # Dec-Feb
        "autumn": [(3, 1),  (5, 31)],   # Mar-May
        "winter": [(6, 1),  (8, 31)],   # Jun-Aug
        "spring": [(9, 1),  (11, 30)],  # Sep-Nov
    }
    months = seasons.get(season, [(1,1),(12,31)])
    start_m, start_d = months[0]
    end_m,   end_d   = months[1]
    if start_m > end_m:  # wraps year (summer)
        yr = today.year if today.month == 12 else today.year - 1
        return date(yr, start_m, start_d), date(yr + 1, end_m, end_d)
    return date(today.year, start_m, start_d), date(today.year, end_m, end_d)

def _last_quarter(today: date) -> tuple:
    q = (today.month - 1) // 3
    if q == 0:
        return date(today.year-1, 10, 1), date(today.year-1, 12, 31)
    starts = [(1,1),(4,1),(7,1),(10,1)]
    ends   = [(3,31),(6,30),(9,30),(12,31)]
    sm, sd = starts[q-1]
    em, ed = ends[q-1]
    return date(today.year, sm, sd), date(today.year, em, ed)

def _last_weekday(text: str, today: date) -> tuple:
    days = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
            "friday":4,"saturday":5,"sunday":6}
    for name, num in days.items():
        if name in text:
            days_back = (today.weekday() - num) % 7 or 7
            d = today - timedelta(days=days_back)
            return d, d
    return today - timedelta(days=7), today

def _month_range(text: str, today: date) -> tuple:
    months = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
              "july":7,"august":8,"september":9,"october":10,"november":11,"december":12}
    for name, num in months.items():
        if name in text:
            import calendar
            _, last = calendar.monthrange(today.year, num)
            return date(today.year, num, 1), date(today.year, num, last)
    return None


# =============================================================================
# SHIFT / TIME-OF-DAY RESOLUTION
# =============================================================================

def resolve_shift(user_msg: str) -> tuple[str | None, str | None]:
    """
    Converts shift/time-of-day expressions to start_datetime/end_datetime.
    Returns (start_datetime, end_datetime) strings or (None, None).
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    lower     = user_msg.lower()

    shift_patterns = {
        # Named shifts
        r'\b(morning shift|morning reports|am shift|morning)\b':
            ("06:00", "11:59"),
        r'\b(afternoon shift|afternoon reports|pm shift|afternoon)\b':
            ("12:00", "17:59"),
        r'\b(evening shift|evening reports)\b':
            ("18:00", "21:59"),
        r'\b(night shift|night reports|overnight|on call|after hours)\b':
            ("22:00", "05:59"),  # crosses midnight — handle below
        r'\b(early morning|before 8|pre-8am)\b':
            ("00:00", "07:59"),
        r'\b(business hours|office hours|9 to 5)\b':
            ("09:00", "17:00"),
        r'\b(out of hours|out-of-hours|ooh)\b':
            ("17:00", "08:59"),  # approximate

        # Weekday / Weekend
        # (these become analytics filters, not datetime range)
    }

    for pattern, (start_t, end_t) in shift_patterns.items():
        if re.search(pattern, lower, re.IGNORECASE):
            return f"{today_str} {start_t}", f"{today_str} {end_t}"

    return None, None


def resolve_weekday_filter(user_msg: str) -> str | None:
    """
    Returns an analytics_intent hint for weekday/weekend filtering.
    """
    lower = user_msg.lower()
    if re.search(r'\bweekend\b', lower):
        return "filter to Saturday and Sunday (visit_datetime.dt.dayofweek >= 5)"
    if re.search(r'\bweekday\b', lower):
        return "filter to Monday-Friday (visit_datetime.dt.dayofweek < 5)"
    if re.search(r'\bmonday\b', lower):
        return "filter to Mondays (visit_datetime.dt.dayofweek == 0)"
    # etc. — could expand for each day
    return None


# =============================================================================
# MULTI-VALUE / OR CONDITIONS
# =============================================================================

def extract_multi_values(intent: dict, user_msg: str, filter_options: dict) -> dict:
    """
    Detects OR conditions and returns them as separate search hints.
    e.g. "CT or MRI" → both modalities should be searched
    Currently expands to search_query so both are captured semantically,
    and flags multi_modality / multi_radiologist for analytics.
    """
    modified = dict(intent)
    lower    = user_msg.lower()

    # Multi-modality: "CT or MRI", "CT and MRI", "CT vs MRI"
    mod_pattern = r'\b(ct|mri|us|cr|nm|mg|xa|rf|pt|dx)\s+(or|and|vs|versus)\s+(ct|mri|us|cr|nm|mg|xa|rf|pt|dx)\b'
    multi_mod = re.findall(mod_pattern, lower, re.IGNORECASE)
    if multi_mod:
        mods = [m[0].upper(), m[2].upper()]
        modified["_multi_modality"] = mods
        modified["modality"]        = None  # Don't restrict to one
        if not modified.get("analytics_intent"):
            modified["analytics_intent"] = f"filter to {' and '.join(mods)}, group by modality, count"
        logging.info(f"[fuzzy] Multi-modality detected: {mods}")

    # Negation: "not CT", "except MRI", "excluding Emergency"
    neg_mod = re.search(r'\b(not|except|excluding|no)\s+(ct|mri|us|cr|nm|mg|xa|rf|pt|dx)\b', lower)
    if neg_mod:
        modified["_exclude_modality"] = neg_mod.group(2).upper()
        logging.info(f"[fuzzy] Exclusion detected: NOT {modified['_exclude_modality']}")

    neg_clinic = re.search(r'\b(not|except|excluding)\s+(emergency|icu|outpatient|surgical|vascular)\b', lower)
    if neg_clinic:
        modified["_exclude_clinic"] = neg_clinic.group(2).upper()
        logging.info(f"[fuzzy] Clinic exclusion: NOT {modified['_exclude_clinic']}")

    return modified


# =============================================================================
# PERFORMANCE EXPRESSION RESOLUTION
# =============================================================================

def resolve_performance_language(intent: dict, user_msg: str) -> dict:
    """
    Maps performance/quality expressions to analytics intents.
    """
    modified = dict(intent)
    lower    = user_msg.lower()

    perf_patterns = {
        r'\b(busiest|most productive|top performer|most reports|highest volume|most active)\b':
            "group by radiologist, count reports, sort descending",
        r'\b(least busy|slowest|lowest volume|fewest reports|least active|bottom performer)\b':
            "group by radiologist, count reports, sort ascending",
        r'\b(above average|better than average|high performers)\b':
            "group by radiologist, count reports, filter above mean",
        r'\b(below average|under-performing|underperform)\b':
            "group by radiologist, count reports, filter below mean",
        r'\b(most experienced|most senior|veteran)\b':
            "group by radiologist, count all-time reports, sort descending",
        r'\b(trending up|increasing|growing)\b':
            "group by month, count reports, sort by month ascending, show trend",
        r'\b(trending down|decreasing|declining|falling)\b':
            "group by month, count reports, sort by month ascending, show trend",
        r'\b(busiest clinic|most active clinic|highest volume clinic)\b':
            "group by clinic, count reports, sort descending",
        r'\b(busiest day|most reports in a day|peak day)\b':
            "group by report_date, count reports, sort descending, head 10",
        r'\b(busiest month|peak month|most active month)\b':
            "group by month (YYYY-MM), count reports, sort descending, head 1",
        r'\b(outlier|unusual|anomaly|anomalous)\b':
            "group by radiologist, count reports, calculate mean and stddev, flag outliers",
    }

    for pattern, analytics_intent in perf_patterns.items():
        if re.search(pattern, lower, re.IGNORECASE):
            if not modified.get("analytics_intent"):
                modified["analytics_intent"] = analytics_intent
                modified["task"]             = "analytics"
                logging.info(f"[fuzzy] Performance expression → analytics: {analytics_intent}")
            break

    return modified


# =============================================================================
# QUANTITY EXPRESSION RESOLUTION
# =============================================================================

def resolve_quantity(intent: dict, user_msg: str) -> dict:
    """Resolves quantity expressions to top_n values."""
    modified = dict(intent)
    lower    = user_msg.lower()

    if re.search(r'\b(a few|handful|couple|just a few)\b', lower):
        modified.setdefault("top_n", 5)
    elif re.search(r'\b(some|several)\b', lower):
        modified.setdefault("top_n", 10)
    elif re.search(r'\b(a lot|many|loads|tonnes|tons|heaps|all of them)\b', lower):
        modified["top_n"] = None  # no limit
    elif re.search(r'\b(everything|all|every single|complete list)\b', lower):
        modified["top_n"] = None

    return modified


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def resolve_intent(intent: dict, user_msg: str, filter_options: dict) -> dict:
    """
    Main fuzzy resolution function. Call this after parse_intent() and
    before run_search() to resolve all fuzzy/ambiguous values.

    Returns an enhanced intent dict with:
      - Exact radiologist name (fuzzy-matched)
      - Exact modality code (alias-resolved)
      - Exact clinic name (fuzzy-matched)
      - Expanded search_query (clinical synonyms)
      - Resolved dates (casual expressions)
      - Analytics intent (performance language)
      - Shift times (start_datetime/end_datetime)
      - Multi-value flags
    """
    resolved = dict(intent)
    resolved["_raw_user_msg"] = user_msg

    available_radiologists = filter_options.get("radiologists", [])
    available_modalities   = filter_options.get("modalities",   [])
    available_clinics      = filter_options.get("clinics",      [])

    # 1. Resolve modality aliases
    if resolved.get("modality"):
        resolved["modality"] = resolve_modality(
            resolved["modality"], available_modalities
        ) or resolved["modality"]

    # 2. Fuzzy-match radiologist name
    if resolved.get("radiologist"):
        resolved["radiologist"] = resolve_radiologist(
            resolved["radiologist"], available_radiologists
        ) or resolved["radiologist"]

    # 3. Fuzzy-match clinic name
    if resolved.get("clinic"):
        resolved["clinic"] = resolve_clinic(
            resolved["clinic"], available_clinics
        ) or resolved["clinic"]

    # 4. Expand clinical synonyms in search_query
    if resolved.get("search_query"):
        resolved["search_query"] = expand_clinical_query(resolved["search_query"])

    # 5. Resolve casual date expressions
    resolved = resolve_relative_dates(resolved)

    # 6. Resolve shift/time-of-day expressions
    if not resolved.get("start_datetime"):
        sdt, edt = resolve_shift(user_msg)
        if sdt:
            resolved["start_datetime"] = sdt
            resolved["end_datetime"]   = edt

    # 7. Resolve performance language
    resolved = resolve_performance_language(resolved, user_msg)

    # 8. Resolve quantity expressions
    resolved = resolve_quantity(resolved, user_msg)

    # 9. Extract multi-value OR conditions
    resolved = extract_multi_values(resolved, user_msg, filter_options)

    # 10. Weekday filter hint
    wd = resolve_weekday_filter(user_msg)
    if wd and not resolved.get("analytics_intent"):
        resolved["analytics_intent"] = wd
        resolved["task"] = "analytics"

    # Clean up internal key
    resolved.pop("_raw_user_msg", None)

    logging.info(f"[fuzzy] Resolved intent: {resolved}")
    return resolved