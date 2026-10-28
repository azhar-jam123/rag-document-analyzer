from pydantic import BaseModel
import json
import ollama

from pathlib import Path
from pypdf import PdfReader
class DocumentAnalysis(BaseModel):
    main_topic: str
    summary: str
    benefits: list[str]
    risks: list[str]
    action_items: list[str]


BASE_DIR = Path(__file__).resolve().parent


filename = input("Enter the document filename: ").strip()

DOCUMENT_PATH = BASE_DIR / filename


if not DOCUMENT_PATH.exists():
    print("Document not found.")
    raise SystemExit


file_extension = DOCUMENT_PATH.suffix.lower()


if file_extension == ".txt":

    with open(
        DOCUMENT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        document_text = file.read()


elif file_extension == ".pdf":

    reader = PdfReader(DOCUMENT_PATH)

    pages_text = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages_text.append(text)

    document_text = "\n".join(pages_text)


else:

    print("Unsupported file type.")
    raise SystemExit


if not document_text.strip():

    print("No readable text found in the document.")
    raise SystemExit


response = ollama.chat(
    model="gemma3:1b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a document analysis assistant. "
                "Extract information accurately from documents. "
                "Return only valid JSON. Do not add explanations."
            )
        },
        {
            "role": "user",
            "content": f"""
Analyze the following document.

Return JSON with exactly these fields:

main_topic
summary
benefits
risks
action_items

Rules:
- main_topic must be a short string
- summary must be a short string
- benefits must be a list
- risks must be a list
- action_items must be a list
- If something is not mentioned, return an empty list

DOCUMENT:

{document_text}
"""
        }
    ],
    format=DocumentAnalysis.model_json_schema()
)


raw_output = response["message"]["content"]

data = DocumentAnalysis.model_validate_json(raw_output)


print("\n--- STRUCTURED DOCUMENT ANALYSIS ---\n")

print(data.model_dump_json(indent=4))