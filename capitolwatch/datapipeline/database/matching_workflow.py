# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

from bs4 import BeautifulSoup

from capitolwatch.services.politician_matcher import match_politician
from capitolwatch.services.reports import update_report_fields
from capitolwatch.services.politicians import get_politician_basic_info
from capitolwatch.db import get_connection
from capitolwatch.datapipeline.database.extractor import (
    extract_politician_name,
    extract_report_year,
)
from config import CONFIG


# NameMatching integration
def setup_namematching():
    """
    Setup NameMatching module by adding it to the Python path.

    Returns:
        The namematching module if available, None otherwise.
    """
    try:
        # Add NameMatching project to Python path
        namematching_path = (
            Path(CONFIG.data_dir).parent.parent / "NameMatching"
        )
        if namematching_path.exists():
            sys.path.insert(0, str(namematching_path))

        import namematching
        return namematching
    except ImportError as e:
        print(f"Warning: NameMatching module not available: {e}")
        return None


def enhanced_name_similarity(name1: str, name2: str) -> float:
    """
    Calculate enhanced name similarity using NameMatching if available,
    fallback to basic string similarity otherwise.

    Args:
        name1: First name to compare
        name2: Second name to compare

    Returns:
        Similarity score between 0.0 and 1.0
    """
    namematching_module = setup_namematching()

    if namematching_module:
        try:
            # Use the AI-powered name matching
            return namematching_module.compare_names(name1, name2)
        except Exception as e:
            print(f"Warning: NameMatching failed, using fallback: {e}")

    # Fallback to basic similarity (Jaro-Winkler-like approach)
    from difflib import SequenceMatcher
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()


def load_manual_overrides() -> Dict[str, str]:
    """
    Load manual overrides from the JSON file.

    Returns:
        Dictionary mapping normalized names to politician IDs.
    """
    override_file = CONFIG.data_dir / "manual_overrides.json"
    try:
        with open(override_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_manual_overrides(overrides: Dict[str, str]) -> None:
    """
    Save manual overrides to the JSON file.

    Args:
        overrides: Dictionary mapping normalized names to politician IDs.
    """
    override_file = CONFIG.data_dir / "manual_overrides.json"
    with open(override_file, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=4, sort_keys=True)


def add_manual_override(name: str, politician_id: str) -> None:
    """
    Add a new manual override and save to file.

    Args:
        name: The normalized name (e.g., "john doe")
        politician_id: The politician ID to map to
    """
    overrides = load_manual_overrides()
    overrides[name.lower().strip()] = politician_id
    save_manual_overrides(overrides)
    print(f"Added manual override: '{name}' → {politician_id}")


def resolve_politician_with_namematching(
    cursor, soup: BeautifulSoup, use_namematching: bool = True
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], float]:
    """
    Enhanced version using NameMatching for validation.

    Returns:
        (politician_id, first_names, last_name, match_type, confidence_score)
    """
    # Get basic resolution first
    politician_id, first_names, last_name, match_type = resolve_politician(
        cursor, soup
    )

    confidence_score = 1.0  # Default confidence for exact matches

    # If we have a match and NameMatching is available, validate it
    if politician_id and use_namematching and match_type != "MANUAL_OVERRIDE":
        try:
            # Get the politician's canonical name from the database
            from capitolwatch.services.politicians import (
                get_politician_basic_info
            )
            politician_info = get_politician_basic_info(
                politician_id, connection=cursor.connection
            )

            if politician_info:
                canonical_name = politician_info["politician_name"]
                extracted_name = f"{first_names} {last_name}"

                # Calculate enhanced similarity
                similarity = enhanced_name_similarity(
                    extracted_name, canonical_name
                )
                confidence_score = similarity

                # If similarity is too low, reject the match
                if similarity < 0.7:  # Threshold for enhanced validation
                    print(
                        f"NameMatching rejected: '{extracted_name}' vs "
                        f"'{canonical_name}' (similarity: {similarity:.3f})"
                    )
                    return None, first_names, last_name, "REJECTED", similarity
                else:
                    print(
                        f"NameMatching validated: '{extracted_name}' vs "
                        f"'{canonical_name}' (similarity: {similarity:.3f})"
                    )
        except Exception as e:
            print(f"Warning: NameMatching validation failed: {e}")

    return politician_id, first_names, last_name, match_type, confidence_score


def resolve_politician(
    cursor, soup: BeautifulSoup
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extract first/last names from an HTML soup and resolve to a politician ID
    using the enhanced matching helper, reusing an existing DB cursor.

    Returns:
        (politician_id, first_names, last_name, match_type)
        -> Any part can be None if extraction fails or no match is found.
    """
    # Extract the politician's first/last names from the page
    first_names, last_name = extract_politician_name(soup)
    if not first_names or not last_name:
        return None, first_names, last_name, None

    # Resolve to a politician ID with match type information
    politician_id, score, confidence, match_type = match_politician(
        cursor, first_names, last_name
    )

    # Only return politician_id if confidence is strong enough
    if confidence in {"HIGH", "MEDIUM"} or match_type == "MANUAL_OVERRIDE":
        return politician_id, first_names, last_name, match_type

    return None, first_names, last_name, match_type


def parse_report_id(filename: str) -> Optional[int]:
    """
    Given a filename like "123.html", return the integer report ID (123).
    Returns None if it cannot be parsed.
    """
    try:
        base = os.path.basename(filename)
        stem = os.path.splitext(base)[0]
        return int(stem)
    except Exception:
        return None


def resolve_report_info_enhanced(
    cursor, soup: BeautifulSoup, use_namematching: bool = True
) -> Tuple[
    Optional[str], Optional[str], Optional[str], Optional[int],
    Optional[str], float
]:
    """
    Enhanced version that includes NameMatching validation.

    Returns:
        (politician_id, first_names, last_name, year, match_type, confidence)
    """
    politician_id, first_names, last_name, match_type, confidence = (
        resolve_politician_with_namematching(cursor, soup, use_namematching)
    )
    year = extract_report_year(soup)
    return politician_id, first_names, last_name, year, match_type, confidence


def process_report_matching(html_file_path: str) -> Optional[str]:
    """
    Parse a stored HTML report, extract the politician name, and resolve it
    to a politician ID using the enhanced matching pipeline.

    Args:
        html_file_path: Path to the HTML report to process.

    Returns:
        The matched politician ID if strong enough; otherwise None.
    """
    conn = get_connection(CONFIG)
    cur = conn.cursor()

    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")

        politician_id, first_names, last_name, match_type = resolve_politician(
            cur, soup
        )

        if not first_names or not last_name:
            print(f"Could not extract name from {html_file_path}")
            return None

        if politician_id:
            match_info = f" ({match_type})" if match_type else ""
            print(
                f"Matched: {first_names} {last_name} → "
                f"{politician_id}{match_info}"
            )
        else:
            print(f"No match: {first_names} {last_name}")

        return politician_id
    finally:
        conn.close()


def main() -> dict:
    """
    Walk through all HTML reports, try to resolve each to a politician ID
    using the enhanced matching helper, and collect simple run statistics.

    Returns:
        {
            "processed": int,
            "matched": int,
            "needs_review": list[{"file": str, "name": str}],
            "manual_overrides_used": int,
        }
    """
    reports_dir = CONFIG.output_folder

    # Load manual overrides for statistics
    manual_overrides = load_manual_overrides()
    print(f"Loaded {len(manual_overrides)} manual overrides")
    if manual_overrides:
        print("Current manual overrides:")
        for name, politician_id in sorted(manual_overrides.items()):
            print(f"  '{name}' → {politician_id}")
        print()

    stats = {
        "processed": 0,
        "matched": 0,
        "updated": 0,
        "needs_review": [],
        "manual_overrides_used": 0,
        "namematching_rejected": 0,
        "confidence_scores": []
    }

    conn = get_connection(CONFIG)
    cur = conn.cursor()

    try:
        for filename in sorted(os.listdir(reports_dir)):
            # Skip non-HTML
            if not filename.endswith(".html"):
                continue

            file_path = os.path.join(reports_dir, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                soup = BeautifulSoup(content, "html.parser")

                # Extract match and metadata with enhanced validation
                (
                    politician_id,
                    first_names,
                    last_name,
                    year,
                    match_type,
                    confidence,
                ) = resolve_report_info_enhanced(cur, soup)

                if first_names and last_name and politician_id:
                    stats["matched"] += 1
                    stats["confidence_scores"].append(confidence)

                    # Count manual overrides used
                    if match_type == "MANUAL_OVERRIDE":
                        stats["manual_overrides_used"] += 1

                    # Update the reports table with the matched politician_id
                    report_id = parse_report_id(filename)
                    if report_id is not None:
                        try:
                            ok = update_report_fields(
                                report_id,
                                politician_id,
                                year=year,
                                connection=conn,
                            )
                            if ok:
                                stats["updated"] += 1
                            else:
                                print(
                                    "Warning: report id "
                                    f"{report_id} not found for "
                                    f"{filename}"
                                )
                        except Exception as e:
                            print(
                                "Error updating report "
                                f"{report_id} for {filename}: {e}"
                            )

                    # Get canonical DB name using the service
                    politician_info = get_politician_basic_info(
                        politician_id, connection=conn
                    )
                    db_name = (
                        politician_info["politician_name"]
                        if politician_info
                        else "?"
                    )

                    print(
                        f"{filename}: {first_names} {last_name} → "
                        f"{politician_id} ({db_name})"
                    )
                elif first_names and last_name:
                    # Check if this was rejected by NameMatching
                    if match_type == "REJECTED":
                        stats["namematching_rejected"] += 1
                        stats["confidence_scores"].append(confidence)

                    stats["needs_review"].append(
                        {
                            "file": filename,
                            "name": f"{first_names} {last_name}",
                            "reason": (
                                match_type if match_type == "REJECTED"
                                else "NO_MATCH"
                            )
                        }
                    )
                    rejection_info = (
                        f" ({match_type})" if match_type == "REJECTED" else ""
                    )
                    print(
                        f"{filename}: {first_names} {last_name} "
                        f"needs review{rejection_info}"
                    )

                stats["processed"] += 1

            except Exception as e:  # keep simple for script usage
                print(f"Error processing {filename}: {e}")

    finally:
        try:
            # Commit all updates if any were made using this connection
            conn.commit()
        except Exception:
            pass
        conn.close()

    print("\nProcessing Summary:")
    print(f"Processed: {stats['processed']}")
    print(f"Matched: {stats['matched']}")
    print(f"Updated: {stats['updated']}")
    print(f"Manual overrides used: {stats['manual_overrides_used']}")
    print(f"NameMatching rejections: {stats['namematching_rejected']}")
    print(f"Need review: {len(stats['needs_review'])}")

    # Calculate average confidence if we have scores
    if stats['confidence_scores']:
        confidence_scores = stats['confidence_scores']
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"Average confidence: {avg_confidence:.3f}")

    return stats


if __name__ == "__main__":
    main()
