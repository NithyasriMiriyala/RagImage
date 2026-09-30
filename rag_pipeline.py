"""Compact Image RAG: image -> OpenCV -> OCR -> chunks -> ChromaDB -> Groq.
CLI:  python rag_pipeline.py --image sample.png [--question "..."]
"""
import argparse, hashlib, os, re, sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the OCR context below (it may have typos). "
            "If the answer is not there, say: \"I couldn't find that in the image.\""),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


class RAGError(Exception):
    """User-facing pipeline error."""


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)


@lru_cache(maxsize=1)
def embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def preprocess(data: bytes, grayscale=True, denoise=True, threshold="otsu", thresh_value=127):
    """Step 1 (OpenCV). threshold: 'otsu' | 'manual' | 'adaptive' | 'none'.
    Fast enough to call on every slider move for a live preview."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RAGError("Could not read the file as an image.")
    if max(img.shape[:2]) < 1500:  # small images OCR badly, so upscale
        s = 1500 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    if denoise:
        img = cv2.GaussianBlur(cv2.medianBlur(img, 3), (3, 3), 0)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if threshold == "otsu":
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold == "manual":
            _, img = cv2.threshold(img, int(thresh_value), 255, cv2.THRESH_BINARY)
        elif threshold == "adaptive":
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 11)
        if threshold != "none" and img.mean() < 127:  # keep text dark on light background
            img = cv2.bitwise_not(img)
    return img


class ImageRAG:
    def __init__(self):
        self.store = None
        self.extracted_text = ""
        self.processed_image = None

    def ingest(self, image_path: str, **opts) -> int:
        """Preprocess -> OCR -> clean -> chunk -> embed into Chroma. Returns chunk count.
        opts are passed to preprocess() (grayscale, denoise, threshold, thresh_value)."""
        path = Path(image_path)
        if not path.is_file():
            raise RAGError(f"Image not found: {image_path}")
        data = path.read_bytes()

        # Step 1: OpenCV
        img = self.processed_image = preprocess(data, **opts)

        # Step 2: OCR
        try:
            raw = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
        except pytesseract.TesseractNotFoundError:
            raise RAGError("Tesseract not installed (brew install tesseract).")

        # Step 3: clean + split
        text = re.sub(r"[ \t]+", " ", raw.replace("\x0c", " "))
        text = re.sub(r"\n{2,}", "\n\n", text).strip()
        if not text:
            raise RAGError("No readable text found. Try different preprocessing settings.")
        self.extracted_text = text
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_text(text)
        docs = [Document(page_content=c, metadata={"chunk": i}) for i, c in enumerate(chunks)]

        # Step 4: embeddings + ChromaDB (collection per image + text, so edits never mix)
        h = hashlib.sha256(data + text.encode()).hexdigest()[:16]
        self.store = Chroma(collection_name=f"img_{h}", embedding_function=embeddings(),
                            persist_directory=CHROMA_DIR)
        self.store.add_documents(docs, ids=[f"{h}-{i}" for i in range(len(docs))])
        return len(docs)

    def ask(self, question: str) -> Answer:
        """Steps 5-6: retrieve top chunks and answer with Groq using only that context."""
        if self.store is None:
            raise RAGError("Ingest an image first.")
        if not question.strip():
            raise RAGError("Question is empty.")
        if not os.getenv("GROQ_API_KEY"):
            raise RAGError("GROQ_API_KEY is not set. Add it to your .env file.")
        docs = self.store.as_retriever(search_kwargs={"k": 4}).invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        chain = PROMPT | ChatGroq(model=GROQ_MODEL, temperature=0) | StrOutputParser()
        try:
            return Answer(chain.invoke({"context": context, "question": question}).strip(), docs)
        except Exception as e:
            raise RAGError(f"Groq API call failed: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--question")
    a = p.parse_args()
    rag = ImageRAG()
    try:
        print(f"Indexed {rag.ingest(a.image)} chunks.")
        if a.question:
            print("Answer:", rag.ask(a.question).text)
            return
        while (q := input("\nQ> ").strip().lower()) not in {"exit", "quit"}:
            if q:
                try:
                    print("A>", rag.ask(q).text)
                except RAGError as e:
                    print("[error]", e)
    except RAGError as e:
        sys.exit(f"Error: {e}")
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()