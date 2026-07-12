
import faiss
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import numpy as np


MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)

def generate_embedding(text: str):
    return model.encode(text, convert_to_numpy=True)


class Retriever:

    def __init__(self, index_path: str, chunks_path: str):

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load chunks (JSON)
        chunks_file = Path(chunks_path)
        with chunks_file.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, prompt: str, top_k: int):

        # Encode user prompt
        query_embedding = generate_embedding(prompt)

        # FAISS expects shape (1, embedding_dim)
        query_vector = np.array(
            [query_embedding],
            dtype=np.float32
        )

        # Normalize if index was built using cosine similarity
        faiss.normalize_L2(query_vector)

        # Similarity search
        scores, indices = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "chunk": self.chunks[idx]
            })

        return results


if __name__ == "__main__":

    prompt = """
    Qui sont Adolphe et Eugène Schneider, et quelles étaient les principales
    activités industrielles de Schneider et Cie au Creusot au XIXe siècle ?
    """

    retriever = Retriever(
        index_path="temp_files\\chunks_index.faiss",
        chunks_path="temp_files\\chunks_reduced.json"
    )

    results = retriever.search(
        prompt=prompt,
        top_k=3
    )

    print("\n===== TOP 3 SIMILAR CHUNKS =====\n")

    for result in results:
        print(f"Rank : {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(result["chunk"])
        print("-" * 80)