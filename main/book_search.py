import time
import sqlite3
import argparse
import numpy as np
from typing import List, Tuple
from lsh_implementation import LSH
from concurrent.futures import ThreadPoolExecutor
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
    try:
        return cursor.execute(sql, (query,)).fetchall()
    except Exception:
        return []

def scan_rows(rows: List[Tuple[str, str, int, bytearray]], query_embedding: np.ndarray, threshold: float=0.3):
    embeddings = np.array([
        np.frombuffer(b_embedding, dtype=query_embedding.dtype) if b_embedding else np.zeros_like(query_embedding)
        for _, _, _, b_embedding in rows
    ])

    scores = query_embedding @ embeddings.T
    above_threshold_indices = np.where(scores >= threshold)[0]
    k = min(5, len(above_threshold_indices))

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
        cursor: sqlite3.Cursor, 
        query_embedding: np.ndarray, 
        threshold: float = 0.3
) -> List[Tuple]:
    cursor.execute("SELECT title, authors, shelf_row, embedding FROM books")
    rows = cursor.fetchall()

    return scan_rows(rows, query_embedding, threshold) if rows else []

def embedding_cluster_search(
        cursor: sqlite3.Cursor, 
        query_embedding: np.ndarray, 
        threshold: float = 0.3
) -> List[Tuple]:
    centroid_id = find_nearest_centroid(cursor, query_embedding)

    if centroid_id == -1:
        return embedding_naive_search(cursor, query_embedding, threshold)

    cursor.execute("SELECT title, authors, shelf_row, embedding FROM books WHERE centroid_id = ?", (centroid_id,))
    rows = cursor.fetchall()
    return scan_rows(rows, query_embedding, threshold) if rows else []

def embedding_lsh_search(
        cursor: sqlite3.Cursor, 
        query_embedding: np.ndarray,
        lsh: LSH,
        threshold: float = 0.3
) -> List[Tuple]:
    hash_keys = lsh.get_hash_keys(cursor, query_embedding)

    search_query = f"""
                    SELECT DISTINCT title, authors, shelf_row, embedding FROM books 
                    JOIN lsh_hash_keys ON books.book_id = lsh_hash_keys.book_id
                    WHERE {" OR ".join("lsh_hash_keys." + col_name + " = ?" for col_name in lsh.table_id_names)}
                    """

    cursor.execute(search_query, hash_keys)
    rows = cursor.fetchall()

    return scan_rows(rows, query_embedding, threshold) if rows else []

def embedding_cluster_lsh_search(
        cursor: sqlite3.Cursor, 
        query_embedding: np.ndarray, 
        lsh: LSH,
        threshold: float = 0.3
) -> List[Tuple]:
    hash_keys = lsh.get_hash_keys(cursor, query_embedding)
    centroid_id = find_nearest_centroid(cursor, query_embedding)

    search_query = f"""
                    SELECT DISTINCT title, authors, shelf_row, embedding FROM books
                    JOIN lsh_hash_keys ON books.book_id = lsh_hash_keys.book_id
                    WHERE books.centroid_id = ? OR ({" OR ".join("lsh_hash_keys." + col_name + " = ?" for col_name in lsh.table_id_names)})
                    """

    cursor.execute(search_query, [centroid_id] + hash_keys)
    rows = cursor.fetchall()

    return scan_rows(rows, query_embedding, threshold) if rows else []

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
        'fts', 'embedding_naive', 'embedding_cluster', 'embedding_lsh', 'embedding_cluster_lsh'
    ], default='embedding_naive')
    parser.add_argument('--similarity_threshold', default=0.3)
    parser.add_argument('--quantize_embedding', action='store_true')

    lsh = LSH()
    times = []
    args = parser.parse_args()
    queries = [
    'what are some books about politics',
    'are there any books that explain how i can get a job',
    'can you recommend books about the political and economic state of the world',
    'i want to start a start up, what should i read',
    'want to get into fashion, what do i read',
    'want to learn more about the french revolution',
    'i am interested in politics, where can i read about it and how it intersects with psychology',
    'are there any fiction books about loss',
    'want to read about time travel',
    'where can i learn about philosophical interpretations of fate',
    'what are some books you can recommend that will help me get into topology',
    'how can i get into stochastic calculus',
    'want to be come a quantitative researcher, how do i begin']

    for query in queries:
        if not args.quantize_embedding:
            embedding = embedding_model.encode(query.lower())[:256]
            embedding = embedding / np.linalg.norm(embedding)
        else:
            embedding = embedding_model.encode(query.lower(), precision = "int8", normalize_embeddings=True)

        start = time.time()
        match args.method:
            case "fts":
                result = fts_search(cursor, query.lower().replace(",", ""))
            case "embedding_naive":
                result = embedding_naive_search(
                    cursor,
                    embedding,
                    float(args.similarity_threshold)
                )
            case "embedding_cluster":
                result = embedding_cluster_search(
                    cursor,
                    embedding,
                    float(args.similarity_threshold)
                )
            case "embedding_lsh":
                result = embedding_lsh_search(
                    cursor,
                    embedding,
                    lsh,
                    float(args.similarity_threshold)
                )
            case "embedding_cluster_lsh":
                result = embedding_cluster_lsh_search(
                    cursor,
                    embedding,
                    lsh,
                    float(args.similarity_threshold)
                )
            case _:
                assert False, 'Invalid method provided'
        for r in result:
            print(r)
        print()
        end = time.time()
        times.append(round(end - start, 3))

    print(f'avg({round(sum(times) / len(times), 3)}) times({times})')

    books_db.close()