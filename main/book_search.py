import sqlite3
import argparse
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

class BookSearch:
    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def fts_search(self, query: str) -> List[Tuple]:
        sql = '''
        SELECT books.id, books.title, books.authors, books.shelf_row, books_fts.rank
        FROM books
        INNER JOIN books_fts ON books.id = books_fts.rowid
        WHERE books_fts.description MATCH ?
        ORDER BY rank
        '''
        return self.cursor.execute(sql, (query,)).fetchall()
    
    def embedding_naive_search(
            self, 
            query: str, 
            embedding_model: SentenceTransformer,
            threshold: float = 0.3, 
            title_weight: float = 0.5,
            description_weight: float = 0.5
    ) -> List[Tuple]:
        self.cursor.execute("SELECT title, authors, shelf_row, title_embedding, description_embedding FROM books")
        rows = self.cursor.fetchall()

        query_embedding = embedding_model.encode(query, prompt_name="query")

        results = []
        for title, authors, row_num, title_embedding_data, description_embedding_data in rows:
            if not title_embedding_data and not description_embedding_data:
                continue
        
            title_embedding = np.frombuffer(title_embedding_data, dtype=np.float32) if title_embedding_data else np.array()
            description_embedding = np.frombuffer(description_embedding_data, dtype=np.float32) if description_embedding_data else np.array()

            title_similarity = self.cosine_similarity(query_embedding, title_embedding) if title_embedding_data else 0
            description_similarity = self.cosine_similarity(query_embedding, description_embedding) if description_embedding_data else 0
            similarity = title_weight * title_similarity + description_weight * description_similarity
            if similarity >= threshold:
                results.append((similarity, title_similarity, description_similarity, title, authors, row_num))

        return list(sorted(results, reverse=True))[:10]

    def close(self) -> None:
        self.connection.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='BookSearch'
            )

    # Run this if you do not have the model saved locally
    # embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")
    # embedding_model.save("../models/snowflake-arctic-embed-m-v1.5")

    embedding_model = SentenceTransformer("../models/finetuned-snowflake-arctic-embed-m-v1.5")

    parser.add_argument('query')
    parser.add_argument('--method', choices=['fts', 'embedding_naive'], default='embedding_naive')
    parser.add_argument('--similarity_threshold', default=0.3)
    parser.add_argument('--title_weight', default=0.5)
    parser.add_argument('--description_weight', default=0.5)

    args = parser.parse_args()

    searcher = BookSearch("../books-database/books.db")

    match args.method:
        case "fts":
            result = searcher.fts_search(args.query)
        case "embedding_naive":
            result = searcher.embedding_naive_search(
                args.query, 
                embedding_model, 
                float(args.similarity_threshold), 
                float(args.title_weight), 
                float(args.description_weight)
            )
        case _:
            assert False

    print("Retrieved books: ")
    for r in result:
        print(r)