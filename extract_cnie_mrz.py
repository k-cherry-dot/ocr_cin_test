import cv2
import pytesseract
import re
from datetime import datetime

# still have to test if this works with pdf or png files

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # path problems with tesseract so i have to manually set this up

def preprocess_mrz_region(image, mrz_height_ratio=0.20):
    """
    Crop the bottom portion of the card (where MRZ lives), convert to grayscale,
    apply a binary threshold, and return the processed image.

    mrz_height_ratio: fraction of image height to treat as MRZ (e.g. bottom 20%).
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
    Run Tesseract OCR on the preprocessed MRZ image.
    We force the whitelist to digits 0-9, uppercase letters A-Z, and '<'.
    We use psm 6 (assume a uniform block of text).
    Then split output into exactly 3 non‐empty lines of 30 characters each.
    """
    custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    raw = pytesseract.image_to_string(processed_img, config=custom_config)

    # normalize line breaks and strip out any other characters
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    # some ocr engines insert short garbage lines; keep only those that look ~30 chars long
    mrz_lines = []
    for ln in lines:
        # remove spaces, since MRZ has no spaces:
        ln_clean = ln.replace(" ", "")
        # if it’s around 30 chars (±2), keep it:
        if 28 <= len(ln_clean) <= 32:
            mrz_lines.append(ln_clean)
    # if more than 3 candidates, take the three longest; if fewer, raise an error.
    mrz_lines = sorted(mrz_lines, key=len, reverse=True)
    if len(mrz_lines) < 3:
        raise ValueError(f"Could not confidently detect 3 MRZ lines. OCR returned: {lines}")
    mrz_lines = mrz_lines[:3]
    # ensure each is exactly 30 chars (pad or truncate if needed)
    mrz_lines = [(ln + "<" * 30)[:30] for ln in mrz_lines]
    return mrz_lines


def parse_mrz_data(line1, line2, line3):
    """
    Given three 30‐char MRZ lines, parse out:
      - card_number (positions 16–23 on line1, eight chars total)
      - date_of_birth (positions 0–5 on line2, as YYYY‐MM‐DD)
      - gender (position 7 on line2)
      - surname & given names from line3

    This version also “cleans up” any stray 'X' → '<' so chevrons are real,
    then takes only the initial contiguous run of A–Z characters for the given name,
    dropping any trailing garbage (like that extra K).
    """
    # 1) normalize any 'x' → '<' in the two lines where they matter:
    #    (tesseract often confuses '<' with 'x', so we restore it.)
    line1 = line1.replace("X", "<")
    line3 = line3.replace("X", "<")

    # 2) card number: take exactly 8 chars from positions [16:24] on line1
    card_number = line1[16:24]

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
    except Exception:
        date_of_birth = f"INVALID({dob_raw})"

    # 4) gender is at line2[7]
    gender = line2[7] if line2[7] in ("M", "F") else "?"

    # 5) name (line3). first drop any trailing '<' filler:
    name_field = line3.rstrip("<")
    if "<<" in name_field:
        surname_raw, given_raw = name_field.split("<<", 1)
    else:
        # if somehow no '<<' was found, treat everything as surname and leave given blank
        surname_raw = name_field
        given_raw = ""

    # 5a) clean surname: replace any '<' → ' ' and strip
    surname = surname_raw.replace("<", " ").strip()

    # 5b) clean given names: only keep the leading a–z run
    #      (if given_raw is e.g. "KENZA<K<<<<", the regex grabs "KENZA" and drops the rest)
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
    High‐level helper: given a filename, load, preprocess, OCR, parse, and return fields.
    """
    # load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")

    # preprocess mrz region (bottom 40%)
    processed_mrz, mrz_crop = preprocess_mrz_region(img, mrz_height_ratio=0.40)

    # save debug files so you can inspect what tesseract sees
    cv2.imwrite("debug_mrz_crop.jpg", mrz_crop)
    cv2.imwrite("debug_mrz_processed.jpg", processed_mrz)

    # ocr the mrz lines
    mrz_lines = ocr_mrz_lines(processed_mrz)
    line1, line2, line3 = mrz_lines
    data = parse_mrz_data(line1, line2, line3)
    return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract_cnie_mrz.py id_card2.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        info = extract_from_image(image_path)
        print("extracted mrz data:")
        for k, v in info.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"error: {e}")
        sys.exit(2)
