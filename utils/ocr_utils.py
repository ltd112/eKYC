import re

# Danh sách cụm từ không mong muốn
unwanted_phrases = [
    "CONG HOA XA HOI CHU NGHIA VIET NAM",
    "SOCIALIST REPUBLIC OF VIET NAM",
    "CAN CUOC CONG DAN",
]

def clean_text(text):
    for unwanted in unwanted_phrases:
        text = text.replace(unwanted, "")
    text = re.sub(r"[^A-Za-z0-9\s\/\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_cccd_content(ocr, image):
    result = ocr.ocr(image, cls=True)
    detected_texts = [clean_text(line[-1][0]) for line in result[0]]
    combined_text = " ".join(detected_texts)
    
    cccd_regex = r"\b\d{9,12}\b"
    name_regex = r"(Ho va ten|Full name)\s+([A-Z]{2,25}(?: [A-Z]{2,25}){0,4})"
    dob_regex = r"\d{2}[\/\-]\d{2}[\/\-]\d{4}"
    
    cccd = None
    name = None
    dob = None

    match_cccd = re.search(cccd_regex, combined_text)
    if match_cccd:
        cccd = match_cccd.group(0)
    
    match_name = re.search(name_regex, combined_text)
    if match_name:
        name = match_name.group(2)
    else:
        fallback_regex = r"\b[A-Z]{2,25}(?: [A-Z]{2,25})*\b"
        fallback_match = re.search(fallback_regex, combined_text)
        if fallback_match:
            extracted_name = fallback_match.group()
            if not any(unwanted in extracted_name for unwanted in unwanted_phrases):
                name = extracted_name

    match_dob = re.search(dob_regex, combined_text)
    if match_dob:
        dob = match_dob.group(0)

    return {"Số CCCD": cccd, "Họ và tên": name, "Ngày sinh": dob}