import fitz  # pymupdf


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


def chunk_text(text, chunk_size=400, overlap=50):
    paragraphs = text.split("\n\n")  # split by paragraphs

    chunks = []
    for para in paragraphs:
        words = para.split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.strip()) > 50:  # avoid tiny chunks
                chunks.append(chunk)

    return chunks


def extract_abstract(text):
    text_lower = text.lower()

    if "abstract" in text_lower:
        start = text_lower.find("abstract")

        # 🔥 increase size
        abstract = text[start:start + 3000]   # was 1500

        return abstract

    return None
