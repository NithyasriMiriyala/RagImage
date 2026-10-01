# Image-Based RAG Question Answering System

You upload an image that has text in it, the app reads the text, and then you can ask questions about it. The answers come only from what's written in the image.

## Abstract

This project is an intelligent question answering system that uses Retrieval-Augmented Generation (RAG) to answer questions from images containing text. The uploaded image is first processed using image preprocessing techniques such as grayscale conversion and thresholding. Optical Character Recognition (OCR) is then used to extract the text from the image.

The extracted text is cleaned, divided into smaller chunks, converted into vector embeddings and stored in a vector database (ChromaDB). When the user asks a question, the RAG system searches the stored information for relevant content and gives it as context to an LLM (Large Language Model). The LLM then generates an answer based on the information extracted from the image.

**Keywords:** Python, OpenCV, OCR, LangChain, ChromaDB, RAG, Groq, LLM

## What is RAG?

RAG stands for Retrieval-Augmented Generation.

An LLM only knows what it learned during training. It can't read your image, and when it doesn't know something it sometimes makes up an answer. RAG fixes that in two steps:

- **Retrieval:** search your own data (here, the text from the image) and find the parts that match the question
- **Generation:** give those parts to the LLM so it writes the answer from them

Think of it like an open book exam. The model isn't answering from memory, it looks at the page you give it and then answers.

## How I built it

1. Upload an image
2. OpenCV cleans it (grayscale, noise reduction, thresholding)
3. Tesseract OCR reads the text from the cleaned image
4. The text is cleaned and split into small chunks using LangChain
5. Each chunk is turned into numbers (an embedding)
6. The embeddings are stored in ChromaDB
7. When you ask a question, ChromaDB finds the closest chunks
8. Those chunks and your question go to the Groq LLM (Llama 3), which writes the answer from that text only

```
image -> OpenCV -> OCR -> chunks -> embeddings -> ChromaDB -> Groq LLM -> answer
```

## Terms used in this project

- **Python:** the language everything is written in
- **OpenCV:** library for image processing
- **Grayscale:** turns a colour image into shades of gray
- **Thresholding:** turns the image into pure black and white so the text stands out
- **OCR:** Optical Character Recognition, reads text from an image
- **Tesseract:** the free OCR engine I used
- **Chunking:** splitting long text into small pieces
- **Embedding:** text converted into numbers that capture its meaning
- **Vector database (ChromaDB):** stores embeddings and finds similar ones
- **LangChain:** framework that connects the splitter, database and LLM
- **Retriever:** the part that searches ChromaDB for the relevant chunks
- **LLM:** Large Language Model, the AI that writes the final answer
- **Groq:** service that runs Llama 3 really fast through an API
- **RAG:** find the relevant text first, then let the LLM answer from it

## What you see when you run the app

**Main page**
- **Input & controls** (left side): upload box and the OpenCV settings
  - Grayscale conversion
  - Noise reduction
  - Thresholding: Otsu (auto), Manual, Adaptive, Off
  - Threshold value (only works with Manual)
  - Extract text button
- **Output** (right side), with two tabs:
  - **Preprocessed image:** original next to the cleaned version, updates live
  - **Extracted text:** the text found, plus Characters, Words and Chunks stored (how many pieces were saved in ChromaDB)

**OpenCV Guide page**
- Explains every setting in simple words, with a small demo

The web page goes up to showing the extracted text. Asking questions is done from the terminal (see below).

## Folder structure

```
app.py                   main page
pages/1_OpenCV_Guide.py  guide page
rag_pipeline.py          the full pipeline
requirements.txt
```

## How to run

Install Tesseract first (it's separate from pip):

```bash
brew install tesseract              # mac
# sudo apt install tesseract-ocr    # ubuntu
```

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To ask questions, get a free key from https://console.groq.com/keys and put it in a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Then run:

```bash
python rag_pipeline.py --image sample.png
```

It gives you a `Q>` prompt in the terminal where you can type questions. Don't push your `.env` file to GitHub, it's already in `.gitignore`.
