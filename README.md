# RAG Practice – Customer FAQ Retriever

A small practice project that retrieves the most relevant answer to a customer's question from a set of FAQ documents, using TF-IDF vectorization and cosine similarity.

## Overview

This is a lightweight, retrieval-only version of RAG (Retrieval-Augmented Generation) — there's no LLM generation step yet, just the **retrieval** half. It takes a user's question, compares it against a small knowledge base of `.txt` FAQ files, and returns the file whose content is most similar to the question.

The knowledge base covers four common customer-support topics:

- `returns.txt` – return policy questions
- `shipping.txt` – shipping questions
- `passwords.txt` – password/account questions
- `gift_wrap.txt` – gift wrapping questions

## How it works

1. **Load documents** – All `.txt` files inside `docs/` are read into memory, paired with their filenames.
2. **Vectorize** – The documents are converted into TF-IDF vectors using `TfidfVectorizer` (with English stop words removed).
3. **Query** – The user's question is typed in at a prompt and transformed into the same TF-IDF vector space.
4. **Match** – Cosine similarity is computed between the question vector and every document vector; the highest-scoring document is returned as the match.
5. **Threshold check** – If the best similarity score is below `MIN_SIMILARITY` (0.25), the script responds that it doesn't have information on that topic instead of returning a low-confidence match.

## Project structure

```
RAG-practice/
├── docs/
│   ├── gift_wrap.txt
│   ├── passwords.txt
│   ├── returns.txt
│   └── shipping.txt
└── retrieve.py
```

## Requirements

- Python 3.8+
- scikit-learn

Install dependencies:

```bash
pip install scikit-learn
```

## Usage

Run the script from the project root:

```bash
python retrieve.py
```

You'll be prompted to type a question. Type `exit` (or `quit`) to end the session.

### Example

```
Indexed 4 documents from docs/.
Type a question, or 'exit' to quit.

Question: how long do I have to return an item?
Matched: returns.txt
Similarity: 0.63

Question: what's the capital of France?
I don't have information on that.
```

## Configuration

- `DOCS_DIR` — folder the script scans for `.txt` FAQ files (defaults to `docs/` next to the script).
- `MIN_SIMILARITY` — minimum cosine similarity score (0–1) required before a match is considered confident enough to return. Currently set to `0.25`.

## Limitations

- Retrieval only — it returns the whole matched FAQ file, not a generated, conversational answer.
- TF-IDF is a simple bag-of-words method, so it can miss matches based on meaning/synonyms rather than exact wording.
- Only returns a single best-matching document per question.

## Possible next steps

- Add an LLM generation step on top of the retrieved document to produce a natural-language answer (full RAG).
- Swap TF-IDF for embeddings (e.g. OpenAI, Sentence-Transformers) for semantic matching.
- Support multi-document context when a question spans more than one topic.
- Add unit tests and a small evaluation set of sample questions.
