import time
import sqlite3
import argparse
import numpy as np
from typing import List, Tuple
from lsh_implementation import LSH
from sentence_transformers import SentenceTransformer
from cluster_embeddings import find_nearest_centroid, cosine_similarity

def fts_search(cursor: sqlite3.Cursor, query: str) -> List[Tuple]:
    sql = '''
    SELECT books.book_id, books.title, books.authors, books.shelf_row, books_fts.rank
    FROM books
    INNER JOIN books_fts ON books.book_id = books_fts.rowid
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
    query_embedding = embedding_model.encode(query, prompt_name="query", normalize_embeddings=True)

    return scan_rows(rows, query_embedding, threshold)

def embedding_cluster_search(
        cursor: sqlite3.Cursor, 
        query: str, 
        embedding_model: SentenceTransformer,
        threshold: float = 0.3,
) -> List[Tuple]:
    query_embedding = embedding_model.encode(query, prompt_name="query", normalize_embeddings=True)
    centroid_id = find_nearest_centroid(cursor, query_embedding)

    if centroid_id == -1:
        print("Cluster table not initialized, so can not run cluster search. Running naive search.")
        return embedding_naive_search(cursor, query, embedding_model, threshold)

    cursor.execute("SELECT title, authors, shelf_row, embedding FROM books WHERE centroid_id = ?", (centroid_id,))
    rows = cursor.fetchall()
    print(len(rows))
    return scan_rows(rows, query_embedding, threshold)

def embedding_lsh_search(
        cursor: sqlite3.Cursor, 
        query: str, 
        embedding_model: SentenceTransformer,
        lsh: LSH,
        threshold: float = 0.3,
) -> List[Tuple]:
    query_embedding = embedding_model.encode(query, prompt_name="query", normalize_embeddings=True)
    hash_keys = lsh.get_hash_keys(cursor, query_embedding)

    search_query = f"""
                    SELECT DISTINCT title, authors, shelf_row, embedding FROM books 
                    JOIN lsh_hash_keys ON books.book_id = lsh_hash_keys.book_id
                    WHERE {" OR ".join("lsh_hash_keys." + col_name + " = ?" for col_name in lsh.table_id_names)}
                    """

    cursor.execute(search_query, hash_keys)
    rows = cursor.fetchall()
    print(len(rows))
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
    parser.add_argument('--method', choices=[
        'fts', 'embedding_naive', 'embedding_cluster', 'embedding_lsh'
    ], default='embedding_naive')
    parser.add_argument('--similarity_threshold', default=0.3)

    lsh = LSH()
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
        case "embedding_lsh":
            result = embedding_lsh_search(
                cursor,
                args.query, 
                embedding_model, 
                lsh,
                float(args.similarity_threshold)
            )
        case _:
            assert False
    for r in result:
        print(r)

    books_db.close()