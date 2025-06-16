import cv2
import pytesseract
import re
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# [Your existing extraction functions remain the same - I'll just show the improved comparison part]

def normalize_date(date_str):
    """
    Normalize various date formats to YYYY-MM-DD
    Handles: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY, etc.
    """
    if not date_str or date_str.strip().upper() in ['', 'NAN', 'NULL', 'NONE']:
        return ""

    date_str = date_str.strip()

    # If already in YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    # Try different date patterns
    patterns = [
        (r'^(\d{2})/(\d{2})/(\d{4})$', lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),  # DD/MM/YYYY
        (r'^(\d{2})/(\d{2})/(\d{2})$', lambda m: f"20{m.group(3)}-{m.group(2)}-{m.group(1)}"),  # DD/MM/YY
        (r'^(\d{4})/(\d{2})/(\d{2})$', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),  # YYYY/MM/DD
        (r'^(\d{2})-(\d{2})-(\d{4})$', lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),  # DD-MM-YYYY
        (r'^(\d{6})$', lambda m: f"20{m.group(1)[4:6]}-{m.group(1)[2:4]}-{m.group(1)[0:2]}"),  # DDMMYY
    ]

    for pattern, converter in patterns:
        match = re.match(pattern, date_str)
        if match:
            try:
                return converter(match)
            except:
                continue

    return date_str  # Return as-is if no pattern matches


def similarity_score(str1, str2):
    """Calculate similarity score between two strings (0-1, where 1 is identical)"""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1.upper(), str2.upper()).ratio()


def compare_with_csv(extracted: dict, csv_path: str, similarity_threshold: float = 0.8):
    """
    Enhanced comparison function with better handling of date formats and fuzzy matching

    Parameters:
        extracted: dict from parse_mrz_data
        csv_path: path to CSV file
        similarity_threshold: minimum similarity score for fuzzy matching (0-1)

    Returns:
        result: dict with detailed comparison results
    """
    # 1) Load the CSV
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        raise RuntimeError(f"Could not read '{csv_path}': {e}")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    required_cols = ["card_number", "surname", "given_names", "date_of_birth", "gender"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in CSV. Found: {list(df.columns)}")

    # 2) Normalize the extracted data
    extracted_norm = {}
    for key, value in extracted.items():
        if key == "date_of_birth":
            extracted_norm[key] = normalize_date(str(value))
        else:
            extracted_norm[key] = str(value).strip().upper() if value else ""

    # 3) Find matching rows by card number (exact match)
    df_clean = df.copy()
    for col in required_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.upper()

    # First try exact card number match
    mask = df_clean["card_number"] == extracted_norm["card_number"]
    matched_rows = df_clean[mask]

    result = {
        "found_row": False,
        "matches": {},
        "similarity_scores": {},
        "expected": {},
        "extracted": extracted_norm,
        "issues_found": []
    }

    if matched_rows.shape[0] == 0:
        # Try fuzzy matching on card number if exact match fails
        best_match_idx = None
        best_score = 0

        for idx, row in df_clean.iterrows():
            score = similarity_score(extracted_norm["card_number"], row["card_number"])
            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match_idx = idx

        if best_match_idx is not None:
            matched_rows = df_clean.iloc[[best_match_idx]]
            result["issues_found"].append(f"Card number fuzzy match (similarity: {best_score:.2f})")
        else:
            result["issues_found"].append("No matching card number found")
            return result

    # Take the first matching row
    row = matched_rows.iloc[0]
    result["found_row"] = True

    # 4) Normalize the expected data from CSV
    expected = {}
    for key in required_cols:
        if key == "date_of_birth":
            expected[key] = normalize_date(row[key])
        else:
            expected[key] = str(row[key]).strip().upper() if pd.notna(row[key]) else ""

    result["expected"] = expected

    # 5) Perform detailed comparison
    for field in required_cols:
        extracted_val = extracted_norm[field]
        expected_val = expected[field]

        # Calculate similarity score
        sim_score = similarity_score(extracted_val, expected_val)
        result["similarity_scores"][field] = sim_score

        # Determine if it's a match (exact or high similarity)
        if extracted_val == expected_val:
            result["matches"][field] = True
        elif sim_score >= similarity_threshold:
            result["matches"][field] = True
            result["issues_found"].append(f"{field}: High similarity match ({sim_score:.2f})")
        else:
            result["matches"][field] = False
            result["issues_found"].append(
                f"{field}: Mismatch - extracted='{extracted_val}' vs expected='{expected_val}'")

    return result


def print_detailed_comparison(comparison_result):
    """Print a detailed comparison report"""
    print("\n" + "=" * 60)
    print("DETAILED COMPARISON REPORT")
    print("=" * 60)

    if not comparison_result["found_row"]:
        print("❌ No matching record found in CSV")
        return

    print("✅ Matching record found in CSV\n")

    # Print field-by-field comparison
    for field in ["card_number", "surname", "given_names", "date_of_birth", "gender"]:
        extracted = comparison_result["extracted"][field]
        expected = comparison_result["expected"][field]
        match = comparison_result["matches"][field]
        similarity = comparison_result["similarity_scores"][field]

        status = "✅ MATCH" if match else "❌ MISMATCH"
        print(f"{status} {field.upper():<12}")
        print(f"  Extracted: '{extracted}'")
        print(f"  Expected:  '{expected}'")
        print(f"  Similarity: {similarity:.2f}")
        print()

    # Print issues found
    if comparison_result["issues_found"]:
        print("ISSUES DETECTED:")
        for issue in comparison_result["issues_found"]:
            print(f"  ⚠️  {issue}")

    # Overall result
    all_match = all(comparison_result["matches"].values())
    print("\n" + "=" * 60)
    if all_match:
        print("🎉 ALL FIELDS MATCH!")
    else:
        match_count = sum(comparison_result["matches"].values())
        total_count = len(comparison_result["matches"])
        print(f"⚠️  {match_count}/{total_count} FIELDS MATCH")
    print("=" * 60)


# Updated main block
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python extract_and_check.py <id_image.jpg> <master_list.csv>")
        sys.exit(1)

    image_path = sys.argv[1]
    csv_path = sys.argv[2]

    try:
        # Extract MRZ data from the ID image
        extracted = extract_from_image(image_path)
        print("EXTRACTED MRZ DATA:")
        for k, v in extracted.items():
            print(f"  {k}: {v}")

        # Compare with CSV using enhanced comparison
        comparison = compare_with_csv(extracted, csv_path, similarity_threshold=0.8)

        # Print detailed comparison report
        print_detailed_comparison(comparison)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(3)