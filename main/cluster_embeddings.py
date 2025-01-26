import sqlite3
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

def cluster(cursor: sqlite3.Cursor, num_clusters):
    cursor.execute("""
    DROP TABLE IF EXISTS book_centroids;
    """)

    cursor.execute("""
    CREATE TABLE book_centroids (
        centroid_id INTEGER PRIMARY KEY,
        centroid BLOB
    );
    """)
    cursor.connection.commit()

    embeddings = []
    book_ids = []

    cursor.execute("SELECT id, embedding FROM books")
    b_embeddings = cursor.fetchall()
    for id, b_embedding in b_embeddings:
        if not b_embedding:
            continue

        book_ids.append(id)
        embeddings.append(np.frombuffer(b_embedding, dtype=np.float32))

    kmeans = KMeans(n_clusters=num_clusters).fit(embeddings)
    centroids = kmeans.cluster_centers_

    for i, centroid in enumerate(centroids):
        blob = centroid.astype(np.float32).tobytes()
        cursor.execute("INSERT INTO book_centroids (centroid_id, centroid) VALUES (?, ?);", (i, blob))

    for book_id, centroid_id in zip(book_ids, kmeans.labels_):
        cursor.execute("UPDATE books SET centroid_id = ? WHERE id = ?;", (int(centroid_id), book_id))

    cursor.connection.commit()

def are_books_clustered(cursor: sqlite3.Cursor) -> bool:
    cursor.execute("""
    SELECT name FROM sqlite_master WHERE type='table' AND name='book_centroids';
    """)

    return bool(cursor.fetchone()[0])

def find_nearest_centroid(cursor: sqlite3.Cursor, embedding: np.ndarray):
    if not are_books_clustered(cursor):
        return -1

    cursor.execute("SELECT centroid_id, centroid FROM book_centroids")
    rows = cursor.fetchall()

    score_row = lambda row: cosine_similarity(embedding, np.frombuffer(row[1], dtype=np.float32)) if row else -1
    return max(rows, key=score_row)[0]

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

if __name__ == '__main__':
    books_db = sqlite3.connect("../books-database/books.db")
    cursor = books_db.cursor()

    cluster(cursor, 5)

    