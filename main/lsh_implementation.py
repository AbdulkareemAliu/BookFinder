import sqlite3
import numpy as np


class LSH:
    def __init__(self, embedding_dimension, cursor: sqlite3.Cursor = None):
        if cursor is None:
            books_db = sqlite3.connect("../books-database/books.db")
            self.cursor = books_db.cursor()
        else:
            self.cursor = cursor

        self.num_planes = 6
        self.num_tables = 8
        self.table_id_names = tuple(f'table_{table_id}_hash_key' for table_id in range(self.num_tables))
        self.embedding_dimension = embedding_dimension

    def reset_tables(self):
        self.cursor.execute("""
                DROP TABLE IF EXISTS lsh_tables;
                """)
        self.cursor.execute("""
                DROP TABLE IF EXISTS lsh_hash_keys;
                """)

        self.cursor.execute("""
                CREATE TABLE lsh_tables (
                    table_id INTEGER PRIMARY KEY,
                    hyperplanes BLOB
                );
                """)

        self.cursor.execute("""
                CREATE TABLE lsh_hash_keys (
                    book_id INTEGER PRIMARY KEY,
                    FOREIGN KEY (book_id) REFERENCES books(book_id)
                );
                """)

        for table_id, table_id_name in enumerate(self.table_id_names):
            b_hyperplanes = np.random.randn(self.num_planes, self.embedding_dimension).astype(np.float32)
            b_hyperplanes /= np.linalg.norm(b_hyperplanes, axis=1, keepdims=True).astype(b_hyperplanes.dtype) + 1e-6
            self.cursor.execute("INSERT INTO lsh_tables (table_id, hyperplanes) VALUES (?, ?);", (table_id, b_hyperplanes))

            self.cursor.execute(f"""
                    ALTER TABLE lsh_hash_keys ADD COLUMN {table_id_name} BLOB;
                    """)

        self.cursor.connection.commit()

    def _hash(self, vector, hyperplanes):
        projections = vector @ hyperplanes.T
        return np.packbits(projections > 0)
    
    def get_hash_keys(self, vector):
        """Returns hash key for each table used for LSH. Assumes that vector is one dimension."""
        self.cursor.execute("""
                SELECT table_id, hyperplanes FROM lsh_tables;
                """)

        hash_keys = [b"" for _ in range(self.num_tables)]
        for table_id, b_hyperplanes in self.cursor.fetchall():
            hyperplanes = np.frombuffer(b_hyperplanes, dtype=np.float32).reshape((self.num_planes, self.embedding_dimension))
            hash_keys[table_id] = self._hash(vector, hyperplanes)

        return hash_keys


    def update(self, book_id: int, vector: np.ndarray):
        """
        Updates LSH tables with input vector.
        Assumes that the vector has NOT already been added.
        """

        hash_keys = tuple(self.get_hash_keys(vector))

        insertion_query = f"INSERT INTO lsh_hash_keys {('book_id',) + self.table_id_names}"
        insertion_query += f" VALUES ({book_id}, " + ', '.join('?' * self.num_tables) + ')'

        self.cursor.execute(insertion_query, hash_keys)
        self.cursor.connection.commit()

    def is_lsh_initialized(self):
        self.cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name='lsh_tables';
        """)

        lsh_tables_initialized = bool(self.cursor.fetchone())

        self.cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name='lsh_hash_keys';
        """)

        lsh_hash_keys_table_initialized = bool(self.cursor.fetchone())

        return lsh_tables_initialized and lsh_hash_keys_table_initialized