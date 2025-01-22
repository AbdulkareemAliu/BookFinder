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
        
    def setup_fts(self) -> None:
        self.cursor.executescript('''
            CREATE VIRTUAL TABLE books_fts USING fts5(
                title, authors, description
            );
                                  
            INSERT INTO books_fts (rowid, title, authors, description)
            SELECT id, title, authors, description FROM books;

            CREATE TRIGGER books_ai AFTER INSERT ON books BEGIN
                INSERT INTO books_fts(rowid, title, authors, description)
                VALUES (new.id, new.title, new.authors, new.description);
            END;

            CREATE TRIGGER books_ad AFTER DELETE ON books BEGIN
                DELETE FROM books_fts
                WHERE rowid = old.id;
            END;

            CREATE TRIGGER books_au AFTER UPDATE ON books BEGIN
                UPDATE books_fts
                SET 
                    title = new.title,
                    authors = new.authors,
                    description = new.description
                WHERE rowid = new.id;
            END;
        ''')
        self.connection.commit()

    def fts_search(self, query: str) -> List[Tuple]:
        sql = '''
        SELECT books.id, books.title, books.authors, books.shelf_row, books_fts.rank
        FROM books
        INNER JOIN books_fts ON books.id = books_fts.rowid
        WHERE books_fts.description MATCH ?
        ORDER BY rank
        '''
        return self.cursor.execute(sql, (query,)).fetchall()
    
    def embedding_search(self, query: str, embedding_model: SentenceTransformer, threshold: int = 0.3) -> List[Tuple]:
        self.cursor.execute("SELECT title, authors, shelf_row, embedding FROM books")
        rows = self.cursor.fetchall()
        
        query_embedding = embedding_model.encode(query)

        results = []
        for title, authors, row_num, b_embedding in rows:
            description_embedding = np.frombuffer(b_embedding, dtype=np.float32)
            similarity = self.cosine_similarity(query_embedding, description_embedding)
            if similarity < threshold:
                continue
            results.append((similarity, title, authors, row_num))

        return list(sorted(results))

    def close(self) -> None:
        self.connection.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='BookSearch'
            )

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    parser.add_argument('query')
    parser.add_argument('--search_method', choices=['fts', 'embedding'], default='embedding')
    parser.add_argument('--initialize_fts', action='store_true')
    parser.add_argument('--similarity_threshold', default=0.3)

    args = parser.parse_args()

    searcher = BookSearch("../books-database/books.db")
    if args.initialize_fts:
        searcher.setup_fts()

    match args.search_method:
        case "fts":
            result = searcher.fts_search(args.query)
        case "embedding":
            result = searcher.embedding_search(args.query, embedding_model, args.similarity_threshold)
        case _:
            assert False


    print(result)