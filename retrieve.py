from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).resolve().parent / "docs"
MIN_SIMILARITY = 0.25


def load_documents(docs_dir: Path) -> tuple[list[str], list[str]]:
    paths = sorted(docs_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")

    filenames: list[str] = []
    texts: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        filenames.append(path.name)
        texts.append(text)

    if not texts:
        raise ValueError(f"All .txt files in {docs_dir} are empty")

    return filenames, texts


def main() -> None:
    filenames, texts = load_documents(DOCS_DIR)
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(texts)

    print(f"Indexed {len(filenames)} documents from {DOCS_DIR}.")
    print("Type a question, or 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            print("Goodbye.")
            break
        if not question:
            continue

        query_vector = vectorizer.transform([question])
        scores = cosine_similarity(query_vector, doc_matrix).flatten()
        best_index = int(scores.argmax())
        best_score = float(scores[best_index])

        if best_score < MIN_SIMILARITY:
            print("I don't have information on that\n")
            continue

        print(f"Matched: {filenames[best_index]}")
        print(f"Similarity: {best_score:.4f}\n")


if __name__ == "__main__":
    main()
