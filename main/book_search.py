import time
import sqlite3
import argparse
import numpy as np
from typing import List, Tuple
from lsh_implementation import LSH
from cluster_embeddings import Clusterer
from sentence_transformers import SentenceTransformer
from calibration_embeddings import get_calibration_embeddings
from sentence_transformers.quantization import quantize_embeddings as quantize
class BookSearchHandler:
    def __init__(
            self,
            search_method: str = "embedding_naive",
            quantize_embeddings: bool = False,
            similarity_threshold: float = 0.3,
            max_num_results: int = 5,
            books_db_path: str="../books-database/books.db",
            embedding_model_path: str="../models/finetuned-snowflake-arctic-embed-m-v1.5",
            model_cache_path: str="./models/cache"
    ):
        self.db = sqlite3.connect(books_db_path)
        self.cursor = self.db.cursor()
        self.embedding_model = SentenceTransformer(embedding_model_path, cache_folder=model_cache_path)

        self.threshold = similarity_threshold
        self.k = max_num_results
        self.quantize_embeddings = quantize_embeddings
        if quantize_embeddings:
            self.calibration_embeddings = get_calibration_embeddings(self.cursor)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        else:
            self.embedding_dimension = 256

        self.lsh = LSH(self.embedding_dimension, self.cursor)
        self.clusterer = Clusterer(self.cursor, should_cache_centroids=True)

        assert search_method in {'fts', 'embedding_naive', 'embedding_cluster', 'embedding_lsh', 'embedding_cluster_lsh'}, "Please enter valid search method"
        self.method = search_method

    def fts_search(self, query: str) -> List[Tuple]:
        sql = '''
        SELECT books.book_id, books.title, books.authors, books.shelf_row, books_fts.rank
        FROM books
        INNER JOIN books_fts ON books.book_id = books_fts.rowid
        WHERE books_fts.description MATCH ?
        ORDER BY rank
        '''
        try:
            return self.cursor.execute(sql, (query,)).fetchall()
        except Exception:
            return []

    def scan_rows(
            self, 
            rows: List[Tuple[str, str, int, bytearray]], 
            query_embedding: np.ndarray
    ):

        embedding_type = np.int8 if self.quantize_embeddings else np.float32
        embeddings = np.array([
            np.frombuffer(b_embedding, dtype=embedding_type) if b_embedding else np.zeros(query_embedding.shape)
            for _, _, _, b_embedding in rows
        ]).astype(query_embedding.dtype)
        embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True).astype(np.float32) + 1e-6)

        scores = query_embedding @ embeddings.T
        above_threshold_indices = np.where(scores >= self.threshold)[0]
        k = min(self.k, len(above_threshold_indices))

        above_threshold_scores = scores[above_threshold_indices]
        top_k_indices = np.argpartition(above_threshold_scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(-above_threshold_scores[top_k_indices])]
        results = [
            (scores[above_threshold_indices[i]], rows[above_threshold_indices[i]][0], 
            rows[above_threshold_indices[i]][1], rows[above_threshold_indices[i]][2])
            for i in top_k_indices
        ]

        return results

    def embedding_naive_search(
            self, 
            query_embedding: np.ndarray,
    ) -> List[Tuple]:
        self.cursor.execute("SELECT title, authors, shelf_row, embedding FROM books")
        rows = self.cursor.fetchall()

        return self.scan_rows(rows, query_embedding) if rows else []

    def embedding_cluster_search(
            self, 
            query_embedding: np.ndarray
    ) -> List[Tuple]:
        centroid_id = self.clusterer.find_nearest_centroid(query_embedding)

        if centroid_id == -1:
            return self.embedding_naive_search(query_embedding)

        self.cursor.execute("SELECT title, authors, shelf_row, embedding FROM books WHERE centroid_id = ?", (centroid_id,))
        rows = self.cursor.fetchall()
        return self.scan_rows(rows, query_embedding) if rows else []

    def embedding_lsh_search(
            self, 
            query_embedding: np.ndarray
    ) -> List[Tuple]:
        hash_keys = self.lsh.get_hash_keys(query_embedding)

        search_query = f"""
                        SELECT DISTINCT title, authors, shelf_row, embedding FROM books 
                        JOIN lsh_hash_keys ON books.book_id = lsh_hash_keys.book_id
                        WHERE {" OR ".join("lsh_hash_keys." + col_name + " = ?" for col_name in self.lsh.table_id_names)}
                        """

        self.cursor.execute(search_query, hash_keys)
        rows = self.cursor.fetchall()

        return self.scan_rows(rows, query_embedding) if rows else []

    def embedding_cluster_lsh_search(
            self, 
            query_embedding: np.ndarray,
    ) -> List[Tuple]:
        hash_keys = self.lsh.get_hash_keys(query_embedding)
        centroid_id = self.clusterer.find_nearest_centroid(query_embedding)

        search_query = f"""
                        SELECT DISTINCT title, authors, shelf_row, embedding FROM books
                        JOIN lsh_hash_keys ON books.book_id = lsh_hash_keys.book_id
                        WHERE books.centroid_id = ? AND ({" OR ".join("lsh_hash_keys." + col_name + " = ?" for col_name in self.lsh.table_id_names)})
                        """

        self.cursor.execute(search_query, [centroid_id] + hash_keys)
        rows = self.cursor.fetchall()

        return self.scan_rows(rows, query_embedding) if rows else []

    def book_search(self, query: str):

        if self.method == 'fts':
            result = self.fts_search(query.lower().replace(",", ""))
        else:
            embedding = self.embedding_model.encode(query.lower())[:self.embedding_dimension]
            if self.quantize_embeddings:
                embedding = quantize(embedding, 'int8', calibration_embeddings=self.calibration_embeddings)
                embedding = embedding.astype(np.float32) / np.linalg.norm(embedding).astype(np.float32)

            match self.method:
                case "embedding_naive":
                    result = self.embedding_naive_search(embedding)
                case "embedding_cluster":
                    result = self.embedding_cluster_search(embedding)
                case "embedding_lsh":
                    result = self.embedding_lsh_search(embedding)
                case "embedding_cluster_lsh":
                    result = self.embedding_cluster_lsh_search(embedding)
                case _:
                    assert False, 'Invalid method provided'

        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='BookSearch'
            )

    parser.add_argument('query')
    parser.add_argument('--method', choices=[
        'fts', 'embedding_naive', 'embedding_cluster', 'embedding_lsh', 'embedding_cluster_lsh'
    ], default='embedding_naive')
    parser.add_argument('--similarity_threshold', default=0.3)
    parser.add_argument('--max_num_results', default=5)
    parser.add_argument('--quantize_embedding', action='store_true')

    args = parser.parse_args()

    search_handler = BookSearchHandler(
        search_method=args.method,
        quantize_embeddings=args.quantize_embedding,
        similarity_threshold=float(args.similarity_threshold),
        max_num_results=int(args.max_num_results)
    )

    result = search_handler.book_search(args.query)
    for r in result:
        print(r)