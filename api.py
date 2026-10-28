import math
import ollama

from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader


app = FastAPI(
    title="AI Document Analyzer",
    description=(
        "Upload TXT or PDF documents, analyze them, "
        "and ask questions using a local LLM and RAG."
    ),
    version="1.0.0"
)


class DocumentAnalysis(BaseModel):
    main_topic: str
    summary: str
    benefits: list[str]
    risks: list[str]
    action_items: list[str]


def analyze_document(document_text: str):

    response = ollama.chat(
        model="gemma3:1b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document analysis assistant. "
                    "Extract information accurately from documents. "
                    "Return only valid structured output."
                )
            },
            {
                "role": "user",
                "content": f"""
Analyze the following document.

Return these fields:

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

    return data


async def extract_text_from_file(file: UploadFile):

    filename = file.filename or ""

    extension = filename.lower().split(".")[-1]

    file_bytes = await file.read()

    if extension == "txt":

        try:
            document_text = file_bytes.decode("utf-8")

        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Could not decode TXT file."
            )

    elif extension == "pdf":

        reader = PdfReader(
            BytesIO(file_bytes)
        )

        pages_text = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages_text.append(text)

        document_text = "\n".join(pages_text)

    else:

        raise HTTPException(
            status_code=400,
            detail="Only TXT and PDF files are supported."
        )

    if not document_text.strip():

        raise HTTPException(
            status_code=400,
            detail="No readable text found in document."
        )

    return document_text


def split_text_into_chunks(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def get_embedding(text: str):

    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    return response["embeddings"][0]


def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (
        magnitude_a * magnitude_b
    )


def retrieve_relevant_chunks(
    document_text: str,
    question: str,
    top_k: int = 3
):

    chunks = split_text_into_chunks(
        document_text
    )

    question_embedding = get_embedding(
        question
    )

    scored_chunks = []

    for index, chunk in enumerate(chunks):

        chunk_embedding = get_embedding(
            chunk
        )

        score = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        scored_chunks.append(
            {
                "chunk_id": index + 1,
                "score": score,
                "text": chunk
            }
        )

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_chunks[:top_k]


@app.get("/")
def home():

    return {
        "message": "AI Document Analyzer API is running"
    }


@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...)
):

    document_text = await extract_text_from_file(file)

    analysis = analyze_document(
        document_text
    )

    return {
        "filename": file.filename,
        "analysis": analysis.model_dump()
    }


@app.post("/ask")
async def ask_document(
    file: UploadFile = File(...),
    question: str = Form(...)
):

    document_text = await extract_text_from_file(file)

    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    relevant_chunks = retrieve_relevant_chunks(
        document_text,
        question,
        top_k=3
    )

    context_parts = []

    for item in relevant_chunks:

        context_parts.append(
            f"""
SOURCE CHUNK {item["chunk_id"]}

{item["text"]}
"""
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    response = ollama.chat(
        model="gemma3:1b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Answer only using the provided context. "
                    "Do not use outside knowledge. "
                    "When possible, mention which source chunk supports the answer. "
                    "If the answer is not present in the context, say: "
                    "'The document does not contain enough information "
                    "to answer this question.'"
                )
            },
            {
                "role": "user",
                "content": f"""
CONTEXT:

{context}

QUESTION:

{question}
"""
            }
        ]
    )

    answer = response["message"]["content"]

    sources = []

    for item in relevant_chunks:

        sources.append(
            {
                "chunk_id": item["chunk_id"],
                "similarity_score": round(
                    float(item["score"]),
                    4
                ),
                "preview": item["text"][:200]
            }
        )

    return {
        "filename": file.filename,
        "question": question,
        "answer": answer,
        "sources": sources
    }