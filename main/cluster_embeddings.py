import sqlite3
import numpy as np
from typing import List
from sklearn.cluster import KMeans

class Clusterer:
    def __init__(self, cursor: sqlite3.Cursor=None, should_cache_centroids: bool = False):
        if cursor is None:
            books_db = sqlite3.connect("../books-database/books.db")
            self.cursor = books_db.cursor()
        else:
            self.cursor = cursor

        self.should_cache_centroids = should_cache_centroids
        self.cached_centroids = None
        self.cached_ids = None

    def cluster(self, num_clusters, embeddings_dtype):
        self.cursor.execute("""
        DROP TABLE IF EXISTS book_centroids;
        """)

        self.cursor.execute("""
        CREATE TABLE book_centroids (
            centroid_id INTEGER PRIMARY KEY,
            centroid BLOB
        );
        """)
        self.cursor.connection.commit()

        embeddings = []
        book_ids = []

        self.cursor.execute("SELECT book_id, embedding FROM books")
        b_embeddings = self.cursor.fetchall()
        for id, b_embedding in b_embeddings:
            if not b_embedding:
                continue

            book_ids.append(id)
            embeddings.append(np.frombuffer(b_embedding, dtype=embeddings_dtype).astype(np.float32))

        kmeans = KMeans(n_clusters=num_clusters).fit(embeddings)
        centroids = kmeans.cluster_centers_

        for i, centroid in enumerate(centroids):
            centroid = (centroid / np.linalg.norm(centroid)).astype(np.float32)
            b_centroid = centroid.tobytes()
            self.cursor.execute("INSERT INTO book_centroids (centroid_id, centroid) VALUES (?, ?);", (i, b_centroid))

        for book_id, centroid_id in zip(book_ids, kmeans.labels_):
            self.cursor.execute("UPDATE books SET centroid_id = ? WHERE book_id = ?;", (int(centroid_id), book_id))

        self.cursor.connection.commit()

    def are_books_clustered(self) -> bool:
        self.cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name='book_centroids';
        """)

        return bool(self.cursor.fetchone())
    
    def _cache(self, centroids: np.ndarray, ids: List[int]):
        if not self.should_cache_centroids:
            return
        self.cached_centroids = centroids
        self.cached_ids = ids


    def find_nearest_centroid(self, embedding: np.ndarray):
        if self.cached_centroids is None:
            self.cursor.execute("SELECT centroid_id, centroid FROM book_centroids")
            rows = self.cursor.fetchall()
            if not rows: return -1
            ids, b_embeddings = zip(*rows)

            centroid_embeddings = np.array([
                np.frombuffer(b_embedding, dtype=np.float32)
                if b_embedding else np.zeros(embedding.shape)
                for b_embedding in b_embeddings
            ]).T

            self._cache(centroid_embeddings, ids)
        else:
            ids = self.cached_ids
            centroid_embeddings = self.cached_centroids

        embedding /= np.linalg.norm(embedding) + 1e-6
        scores = embedding @ centroid_embeddings

        return ids[np.argmax(scores)]

if __name__ == '__main__':
    clusterer = Clusterer()
    clusterer.cluster(5, np.float32)