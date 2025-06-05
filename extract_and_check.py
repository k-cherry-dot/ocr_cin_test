import cv2
import pytesseract
import re
import pandas as pd
from datetime import datetime

# problem with tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# extracting

def preprocess_mrz_region(image, mrz_height_ratio=0.40):
    """
    Crop the bottom portion of the card (where the MRZ lives), upscale it,
    convert to grayscale, apply blur + adaptive threshold, and invert so that
    characters are dark on a light background for Tesseract.
    """
    h, w = image.shape[:2]
    mrz_start_y = int(h * (1 - mrz_height_ratio))
    mrz_crop = image[mrz_start_y : h, 0 : w]

    # upscale by 2× to improve ocr accuracy on small mrz characters
    mrz_crop = cv2.resize(
        mrz_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # adaptive thresholding; adjust c if needed
    th = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=15
    )

    processed = cv2.bitwise_not(th)
    return processed, mrz_crop


def ocr_mrz_lines(processed_img):
    """
    Run Tesseract OCR on the preprocessed MRZ image with a whitelist of
    A–Z, 0–9, and '<'. Use PSM 6 (assume a uniform block of text). Then:
      1) Split the raw output into nonempty lines
      2) Remove spaces from each line
      3) Sort all candidates by length (descending)
      4) Take the top 3 lines, pad or truncate each to exactly 30 chars
    """
    custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    raw = pytesseract.image_to_string(processed_img, config=custom_config)

    # split on newlines, strip whitespace, discard empty strings
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    if len(lines) < 3:
        raise ValueError(f"Could not detect at least 3 text lines. OCR returned: {lines}")

    # remove any spaces, sort by length descending, take top 3
    cleaned = [ln.replace(" ", "") for ln in lines]
    candidates = sorted(cleaned, key=len, reverse=True)[:3]

    # ensure each line is exactly 30 chars: pad with '<' or truncate
    mrz_lines = [(ln + "<" * 30)[:30] for ln in candidates]
    return mrz_lines


def parse_mrz_data(line1, line2, line3):
    """
    Given three 30-character MRZ lines, extract:
      - card_number (8 chars at positions [16:24] in line1)
      - date_of_birth (YYMMDD at line2[0:6], converted to "YYYY-MM-DD")
      - gender (line2[7])
      - surname & given_names (from line3, splitting on "<<")
    Also normalizes any stray 'X' back to '<' before splitting the name.
    """
    # 1) normalize ocr's 'x' → '<' (chevrons)
    line1 = line1.replace("X", "<")
    line3 = line3.replace("X", "<")

    # 2) card number: take exactly 8 chars from positions [16:24] of line1
    card_number = line1[16:24]

    # 3) date of birth (YYMMDD → YYYY-MM-DD)
    dob_raw = line2[0:6]
    try:
        yy = int(dob_raw[0:2])
        if yy <= 25:
            year = 2000 + yy
        else:
            year = 1900 + yy
        month = int(dob_raw[2:4])
        day = int(dob_raw[4:6])
        date_of_birth = datetime(year, month, day).strftime("%Y-%m-%d")
    except:
        date_of_birth = f"INVALID({dob_raw})"

    # 4) gender at line2[7]
    gender = line2[7] if line2[7] in ("M", "F") else "?"

    # 5) name (line3): split on "<<"
    name_field = line3.rstrip("<")
    if "<<" in name_field:
        surname_raw, given_raw = name_field.split("<<", 1)
    else:
        surname_raw, given_raw = name_field, ""

    # 5a) clean surname: replace '<' → ' ' and strip
    surname = surname_raw.replace("<", " ").strip()

    # 5b) clean given names: keep only the initial run of a–z
    m = re.match(r"^([A-Z]+)", given_raw)
    if m:
        given_names = m.group(1)
    else:
        given_names = ""

    return {
        "card_number":    card_number,
        "date_of_birth":  date_of_birth,
        "gender":         gender,
        "surname":        surname,
        "given_names":    given_names
    }


def extract_from_image(image_path):
    """
    High-level helper that:
      1) Loads the image from disk
      2) Crops & preprocesses the MRZ region
      3) Runs OCR to get three MRZ lines
      4) Parses those lines into a dict with keys:
         { card_number, date_of_birth, gender, surname, given_names }
    Also writes out debug images: debug_mrz_crop.jpg and debug_mrz_processed.jpg.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")

    processed_mrz, mrz_crop = preprocess_mrz_region(img, mrz_height_ratio=0.40)

    # save debug files so you can inspect what tesseract sees
    cv2.imwrite("debug_mrz_crop.jpg", mrz_crop)
    cv2.imwrite("debug_mrz_processed.jpg", processed_mrz)

    mrz_lines = ocr_mrz_lines(processed_mrz)
    line1, line2, line3 = mrz_lines
    return parse_mrz_data(line1, line2, line3)


# comparison
# this doesnt work yet
# have to try with cvs files

def compare_with_excel(extracted: dict, excel_path: str, sheet_name: str = None):
    """
    Compare 'extracted' (the dict returned by parse_mrz_data) against a record in Excel.

    Parameters:
      extracted  : dict
                   { "card_number": ..., "surname": ..., "given_names": ...,
                     "date_of_birth": ..., "gender": ... }
      excel_path : str
                   Path to the .xlsx file containing columns:
                     card_number | surname | given_names | date_of_birth | gender
      sheet_name : str or None
                   If your workbook has multiple sheets, specify the sheet name. Otherwise None.

    Returns:
      result : dict with keys:
        - found_row : bool (True if a row with matching card_number was found)
        - matches   : { field_name: True/False, … }
        - expected  : { card_number, surname, given_names, date_of_birth, gender } (from Excel)
        - extracted : same as the normalized 'extracted' input
    """
    # 1) load the excel sheet into a dataframe
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=str, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"Could not read '{excel_path}': {e}")

    # 2) normalize (uppercase & strip) the five key columns in the dataframe
    for col in ["card_number", "surname", "given_names", "date_of_birth", "gender"]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in Excel. Found: {list(df.columns)}")
        df[col] = df[col].astype(str).str.strip().str.upper()

    # 3) normalize the 'extracted' dict in the same way
    extracted_norm = {
        "card_number":   extracted["card_number"].strip().upper(),
        "surname":       extracted["surname"].strip().upper(),
        "given_names":   extracted["given_names"].strip().upper(),
        "date_of_birth": extracted["date_of_birth"].strip(),    # already "YYYY-MM-DD"
        "gender":        extracted["gender"].strip().upper(),
    }

    # 4) find rows where card_number matches
    mask = df["card_number"] == extracted_norm["card_number"]
    matched = df[mask]

    result = {
        "found_row": False,
        "matches":   {},
        "expected":  {},
        "extracted": extracted_norm
    }

    if matched.shape[0] == 0:
        # no matching row found
        return result

    # take the first matching row if there are duplicates
    row = matched.iloc[0]
    result["found_row"] = True

    expected = {
        "card_number":   row["card_number"],
        "surname":       row["surname"],
        "given_names":   row["given_names"],
        "date_of_birth": row["date_of_birth"],
        "gender":        row["gender"]
    }
    result["expected"] = expected

    # 5) do a strict string comparison for each field
    for key in ["card_number", "surname", "given_names", "date_of_birth", "gender"]:
        result["matches"][key] = (extracted_norm[key] == expected[key])

    return result


# main block

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python extract_and_check.py <id_image.jpg> <master_list.xlsx>")
        sys.exit(1)

    image_path = sys.argv[1]
    excel_path = sys.argv[2]

    try:
        # 1) extract mrz data from the id image
        extracted = extract_from_image(image_path)
        print("extracted mrz data:")
        for k, v in extracted.items():
            print(f"  {k}: {v}")

        # 2) compare the extracted data against the excel file
        comparison = compare_with_excel(extracted, excel_path)
        if not comparison["found_row"]:
            print(f"\n no row in '{excel_path}' matched card_number = {extracted['card_number']}")
            sys.exit(2)

        print("\n comparison against excel:")
        for field in ["card_number", "surname", "given_names", "date_of_birth", "gender"]:
            got = comparison["extracted"][field]
            exp = comparison["expected"][field]
            ok  = comparison["matches"][field]
            mark = "ok" if ok else "not ok"
            print(f"  {mark} {field:12s}: extracted='{got}'    expected='{exp}'")

        if all(comparison["matches"].values()):
            print("\n all fields match!")
        else:
            print("\n some fields did not match. please review above.")

    except Exception as e:
        print(f"error: {e}")
        sys.exit(3)