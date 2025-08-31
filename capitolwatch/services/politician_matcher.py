# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import json
from typing import Dict, Optional, Tuple

from capitolwatch.db import get_connection
from capitolwatch.services.politicians import get_politician_id, normalize_name
from config import CONFIG

# Prefer external compare_names for better accuracy; fallback if unavailable
try:  # pragma: no cover - optional dep
    from namematching import compare_names  # type: ignore
except Exception:  # pragma: no cover - optional dep
    compare_names = None


def fallback_compare_names(a: str, b: str) -> float:
    """Lightweight token Jaccard similarity with a small last-name bonus."""
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)
    if a_norm == b_norm:
        return 1.0

    a_tokens = a_norm.split()
    b_tokens = b_norm.split()
    if not a_tokens or not b_tokens:
        return 0.0

    set_a, set_b = set(a_tokens), set(b_tokens)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    base = inter / union if union else 0.0
    last_bonus = 0.1 if (a_tokens[-1] == b_tokens[-1]) else 0.0
    return min(1.0, base + last_bonus)


def score_names(report_name: str, db_name: str) -> float:
    """Score similarity between a normalized report name and a DB name."""
    if compare_names is not None:
        return float(compare_names(report_name, db_name.lower()))
    return fallback_compare_names(report_name, db_name)


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "HIGH": 0.95,
    "MEDIUM": 0.85,
    "LOW": 0.70,
}


def load_manual_overrides() -> Dict[str, str]:
    """Load overrides from disk once and cache them."""
    override_file = CONFIG.data_dir / "manual_overrides.json"
    try:
        with open(override_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    # Normalize keys for stable lookup
    return {normalize_name(k): v for k, v in data.items()}


def confidence_for(score: float) -> str:
    if score >= DEFAULT_THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= DEFAULT_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    if score >= DEFAULT_THRESHOLDS["LOW"]:
        return "LOW"
    return "VERY_LOW"


def match_politician(
    cur,
    first_names: str,
    last_name: str,
) -> Tuple[Optional[str], float, str, str]:
    """
    Try to match a politician from a (first_names, last_name) input.

    Returns:
        Tuple of:
            - politician_id (str | None): The matched politician ID, or None
            - score (float): Similarity score (0–1)
            - confidence (str): Confidence level
                ("HIGH", "MEDIUM", "LOW", "VERY_LOW", "NO_MATCH")
            - match_type (str): Origin of the match
                ("MANUAL_OVERRIDE", "AUTOMATIC", "NO_MATCH")
    """
    # Load manual overrides
    overrides = load_manual_overrides()

    first_names_norm = normalize_name(first_names)
    last_name_norm = normalize_name(last_name)
    report_name = f"{first_names_norm} {last_name_norm}".strip()

    # 1) Manual override
    if report_name in overrides:
        return overrides[report_name], 1.0, "HIGH", "MANUAL_OVERRIDE"

    # 2) Exact DB lookup
    connection = getattr(cur, "connection", None)
    politician_id = get_politician_id(
        first_names_norm, last_name_norm, connection=connection
    )

    if politician_id:
        # Exact match found
        return politician_id, 1.0, "HIGH", "AUTOMATIC"

    # 3) Similarity-based fallback search (no exact match)
    should_close = False
    if connection is None:
        connection, should_close = get_connection(CONFIG), True
    try:
        cursor = connection.cursor()

        # First pass: restrict candidate set by normalized last name
        cursor.execute(
            """
            SELECT id, first_name, last_name
            FROM politicians
            WHERE lower(last_name) = ?
            """,
            (last_name_norm,),
        )
        candidates = cursor.fetchall() or []

        # If nothing found (or no index), fallback to scanning all rows
        if not candidates:
            cursor.execute(
                "SELECT id, first_name, last_name FROM politicians"
            )
            candidates = cursor.fetchall() or []
    finally:
        if should_close:
            connection.close()

    # Rank candidates by similarity to the normalized report name
    best_id = None
    best_score = 0.0
    for row in candidates:
        if isinstance(row, dict) or hasattr(row, "keys"):
            candidate_id = row["id"]
            db_first, db_last = row["first_name"], row["last_name"]
        else:
            candidate_id, db_first, db_last = row  # tuple fallback
        db_name = f"{db_first} {db_last}"

        score = score_names(report_name, db_name)
        if score > best_score:
            best_score = score
            best_id = candidate_id

    if best_id:
        confidence = confidence_for(best_score)
        return best_id, best_score, confidence, "AUTOMATIC"

    # 4) No match
    return None, 0.0, "NO_MATCH", "NO_MATCH"


def get_politician_id_by_name_enhanced(
    cursor,
    first_name_tokens: list[str],
    last_name_tokens: list[str],
) -> Optional[str]:
    """
    Wrapper around match_politician that only returns IDs for strong matches.

    Args:
        cursor: SQLite cursor
        first_name_tokens (list[str]): Tokens the first name(s)
        last_name_tokens (list[str]): Tokens the last name(s)

    Returns:
        str | None:
            - Politician ID if confidence is HIGH or MEDIUM,
              or if resolved via manual override.
            - None otherwise (weak or no match).
    """
    first_names = " ".join(first_name_tokens)
    last_name = " ".join(last_name_tokens)

    politician_id, score, confidence, match_type = match_politician(
        cursor, first_names, last_name
    )

    if confidence in {"HIGH", "MEDIUM"} or match_type == "MANUAL_OVERRIDE":
        return politician_id
    return None
