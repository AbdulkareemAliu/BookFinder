import time
import sqlite3
import argparse
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from cluster_embeddings import find_nearest_centroid, cosine_similarity

def fts_search(cursor: sqlite3.Cursor, query: str) -> List[Tuple]:
    sql = '''
    SELECT books.id, books.title, books.authors, books.shelf_row, books_fts.rank
    FROM books
    INNER JOIN books_fts ON books.id = books_fts.rowid
    WHERE books_fts.description MATCH ?
    ORDER BY rank
    '''
    return cursor.execute(sql, (query,)).fetchall()

def scan_rows(rows: List[Tuple], query_embedding: np.ndarray, threshold: float=0.3, k: int=10):

    results = []
    for title, authors, row_num, b_embedding in rows:
        if not b_embedding:
            continue
    
        embedding = np.frombuffer(b_embedding, dtype=np.float32) if b_embedding else np.array()
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity >= threshold:
            results.append((similarity, title, authors, row_num))

    return list(sorted(results, reverse=True))[:k]


def embedding_naive_search(
        cursor: sqlite3.Cursor, 
        query: str, 
        embedding_model: SentenceTransformer,
        threshold: float = 0.3,
) -> List[Tuple]:
    cursor.execute("SELECT title, authors, shelf_row, embedding FROM books")
    rows = cursor.fetchall()
    query_embedding = embedding_model.encode(query, prompt_name="query")

    return scan_rows(rows, query_embedding, threshold)

def embedding_cluster_search(
        cursor: sqlite3.Cursor, 
        query: str, 
        embedding_model: SentenceTransformer,
        threshold: float = 0.3,
) -> List[Tuple]:
    query_embedding = embedding_model.encode(query, prompt_name="query")
    centroid_id = find_nearest_centroid(cursor, query_embedding)

    if centroid_id == -1:
        print("Cluster table not initialized, so can not run cluster search. Running naive search.")
        return embedding_naive_search(cursor, query, embedding_model, threshold)

    cursor.execute("SELECT title, authors, shelf_row, embedding FROM books WHERE centroid_id = ?", (centroid_id,))
    rows = cursor.fetchall()
    return scan_rows(rows, query_embedding, threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='BookSearch'
            )
    
    books_db = sqlite3.connect("../books-database/books.db")
    cursor = books_db.cursor()

    # Run this if you do not have the model saved locally
    # embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")
    # embedding_model.save("../models/snowflake-arctic-embed-m-v1.5")

    embedding_model = SentenceTransformer("../models/finetuned-snowflake-arctic-embed-m-v1.5", cache_folder="./models/cache")

    parser.add_argument('query')
    parser.add_argument('--method', choices=['fts', 'embedding_naive', 'embedding_cluster'], default='embedding_naive')
    parser.add_argument('--similarity_threshold', default=0.3)

    args = parser.parse_args()
    match args.method:
        case "fts":
            result = fts_search(cursor, args.query)
        case "embedding_naive":
            result = embedding_naive_search(
                cursor,
                args.query, 
                embedding_model, 
                float(args.similarity_threshold)
            )
        case "embedding_cluster":
            result = embedding_cluster_search(
                cursor,
                args.query, 
                embedding_model, 
                float(args.similarity_threshold)
            )
        case _:
            assert False
    for r in result:
        print(r)

    books_db.close()