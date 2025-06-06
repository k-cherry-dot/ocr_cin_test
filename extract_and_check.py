import cv2
import pytesseract
import re
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# path problems with tesseract so i have to manually set this up

# extraction part (unchanged)

def preprocess_mrz_region(image, mrz_height_ratio=0.20):
    """
    crop the bottom portion of the card (where mrz lives), convert to grayscale,
    apply a binary threshold, and return the processed image.

    mrz_height_ratio: fraction of image height to treat as mrz (e.g. bottom 20%).
    """
    h, w = image.shape[:2]
    mrz_start_y = int(h * (1 - mrz_height_ratio))
    mrz_crop = image[mrz_start_y: h, 0: w]

    gray = cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2GRAY)
    # apply a moderate gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # adaptive threshold (works well with varying lighting)
    th = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=15
    )

    # invert back so characters are dark on light if needed (pytesseract prefers dark-on-light)
    processed = cv2.bitwise_not(th)
    return processed, mrz_crop


def ocr_mrz_lines(processed_img):
    """
    run tesseract ocr on the preprocessed mrz image.
    we force the whitelist to digits 0-9, uppercase letters a-z, and '<'.
    we use psm 6 (assume a uniform block of text).
    then split output into exactly 3 non-empty lines of ~30 characters each.
    """
    custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    raw = pytesseract.image_to_string(processed_img, config=custom_config)

    # normalize line breaks and strip out any other characters
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    # some ocr engines insert short garbage lines; keep only those that look ~30 chars long
    mrz_lines = []
    for ln in lines:
        # remove spaces, since mrz has no spaces:
        ln_clean = ln.replace(" ", "")
        # if it’s around 30 chars (±2), keep it:
        if 28 <= len(ln_clean) <= 32:
            mrz_lines.append(ln_clean)
    # if more than 3 candidates, take the three longest; if fewer, raise an error.
    mrz_lines = sorted(mrz_lines, key=len, reverse=True)
    if len(mrz_lines) < 3:
        raise ValueError(f"could not confidently detect 3 mrz lines. ocr returned: {lines}")
    mrz_lines = mrz_lines[:3]
    # ensure each is exactly 30 chars (pad or truncate if needed)
    mrz_lines = [(ln + "<" * 30)[:30] for ln in mrz_lines]
    return mrz_lines


def parse_mrz_data(line1, line2, line3):
    """
    given three 30-char mrz lines, parse out:
      - card_number (positions 16–23 on line1, eight chars total)
      - date_of_birth (positions 0–5 on line2, as yyyy-mm-dd)
      - gender (position 7 on line2)
      - surname & given_names from line3

    this version also “cleans up” any stray 'x' → '<' so chevrons are real,
    then takes only the initial contiguous run of a–z characters for the given name,
    dropping any trailing garbage.
    """
    # 1) normalize ocr's 'x' → '<' (chevrons)
    line1 = line1.replace("X", "<")
    line3 = line3.replace("X", "<")

    # 2) card number: take exactly 8 chars from positions [16:24] of line1, then strip '<'
    raw_card = line1[16:24]
    card_number = raw_card.replace("<", "")

    # 3) date of birth (yy mm dd → yyyy-mm-dd)
    dob_raw = line2[0:6]  # e.g. "040513"
    try:
        yy = int(dob_raw[0:2])
        # assume anything 00–25 → 2000+, else 1900+
        if yy <= 25:
            year = 2000 + yy
        else:
            year = 1900 + yy
        month = int(dob_raw[2:4])
        day = int(dob_raw[4:6])
        date_of_birth = datetime(year, month, day).strftime("%Y-%m-%d")
    except:
        date_of_birth = f"invalid({dob_raw})"

    # 4) gender is at line2[7]
    gender = line2[7] if line2[7] in ("M", "F") else "?"

    # 5) name (line3). first drop any trailing '<' filler:
    name_field = line3.rstrip("<")
    if "<<" in name_field:
        surname_raw, given_raw = name_field.split("<<", 1)
    else:
        surname_raw, given_raw = name_field, ""

    # 5a) clean surname: replace '<' → ' ' and strip
    surname = surname_raw.replace("<", " ").strip()

    # 5b) clean given names: only keep the leading a–z run
    m = re.match(r"^([A-Z]+)", given_raw)
    if m:
        given_names = m.group(1)
    else:
        given_names = ""  # if no initial run of letters, just empty

    return {
        "card_number":    card_number,
        "date_of_birth":  date_of_birth,
        "gender":         gender,
        "surname":        surname,
        "given_names":    given_names
    }


def extract_from_image(image_path):
    """
    high-level helper: given a filename, load, preprocess, ocr, parse, and return fields.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"could not load image at '{image_path}'")

    processed_mrz, mrz_crop = preprocess_mrz_region(img, mrz_height_ratio=0.40)

    # save debug files so you can inspect what tesseract sees
    cv2.imwrite("debug_mrz_crop.jpg", mrz_crop)
    cv2.imwrite("debug_mrz_processed.jpg", processed_mrz)

    mrz_lines = ocr_mrz_lines(processed_mrz)
    line1, line2, line3 = mrz_lines
    data = parse_mrz_data(line1, line2, line3)
    return data


# comparison part (uses csv) (works now)

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

    # diff date patterns
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

    return date_str  #returns as is if there are no matches


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
    # 1) loading the csv file
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        raise RuntimeError(f"Could not read '{csv_path}': {e}")

    # stripping whitespace (added this after leaving a whitespace in one of my columns)
    df.columns = df.columns.str.strip()

    required_cols = ["card_number", "surname", "given_names", "date_of_birth", "gender"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in CSV. Found: {list(df.columns)}")

    # 2) normalizing the extracted data
    extracted_norm = {}
    for key, value in extracted.items():
        if key == "date_of_birth":
            extracted_norm[key] = normalize_date(str(value))
        else:
            extracted_norm[key] = str(value).strip().upper() if value else ""

    # 3) find matching rows by card number (exact match) (this is bcs in the csv file, there will be a loooot of info)
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
        # try fuzzy matching on card number if exact match fails (added this after having some trouble)
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

    # take the first matching row
    row = matched_rows.iloc[0]
    result["found_row"] = True

    # 4) normalize the expected data from CSV
    expected = {}
    for key in required_cols:
        if key == "date_of_birth":
            expected[key] = normalize_date(row[key])
        else:
            expected[key] = str(row[key]).strip().upper() if pd.notna(row[key]) else ""

    result["expected"] = expected

    # 5) perform detailed comparison
    for field in required_cols:
        extracted_val = extracted_norm[field]
        expected_val = expected[field]

        # Calculate similarity score
        sim_score = similarity_score(extracted_val, expected_val)
        result["similarity_scores"][field] = sim_score

        # determine if it's a match (exact or high similarity) (the treshold is 0.8)
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
    print("DETAILED COMPARISON REPORT")

    if not comparison_result["found_row"]:
        print("No matching record found in CSV")
        return

    print("Matching record found in CSV\n")

    # print field-by-field comparison
    for field in ["card_number", "surname", "given_names", "date_of_birth", "gender"]:
        extracted = comparison_result["extracted"][field]
        expected = comparison_result["expected"][field]
        match = comparison_result["matches"][field]
        similarity = comparison_result["similarity_scores"][field]

        status = "MATCH" if match else "MISMATCH"
        print(f"{status} {field.upper():<12}")
        print(f"  Extracted: '{extracted}'")
        print(f"  Expected:  '{expected}'")
        print(f"  Similarity: {similarity:.2f}")
        print()

    # print issues found (for debugging purposes)
    if comparison_result["issues_found"]:
        print("Issued Detected:")
        for issue in comparison_result["issues_found"]:
            print(f" {issue}")

    # overall result
    all_match = all(comparison_result["matches"].values())
    if all_match:
        print("All fields match")
    else:
        match_count = sum(comparison_result["matches"].values())
        total_count = len(comparison_result["matches"])
        print(f"{match_count}/{total_count} FIELDS MATCH")


# main block
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python extract_and_check.py <id_image.jpg> <master_list.csv>")
        sys.exit(1)

    image_path = sys.argv[1]
    csv_path = sys.argv[2]

    try:
        # extract MRZ data from the ID image
        extracted = extract_from_image(image_path)
        print("EXTRACTED MRZ DATA:")
        for k, v in extracted.items():
            print(f"  {k}: {v}")

        # compare with CSV using enhanced comparison
        comparison = compare_with_csv(extracted, csv_path, similarity_threshold=0.8)

        # print detailed comparison report
        print_detailed_comparison(comparison)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(3)