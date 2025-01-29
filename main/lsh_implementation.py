import sqlite3
import numpy as np


class LSH:
    def __init__(self, embedding_type: np.dtype = np.float32):
        self.num_planes = 6
        self.num_tables = 8
        self.plane_type = embedding_type
        self.table_id_names = tuple(f'table_{table_id}_hash_key' for table_id in range(self.num_tables))

    def reset_tables(self, cursor: sqlite3.Cursor, embedding_dimension):
        cursor.execute("""
                DROP TABLE IF EXISTS lsh_tables;
                """)
        cursor.execute("""
                DROP TABLE IF EXISTS lsh_hash_keys;
                """)

        cursor.execute("""
                CREATE TABLE lsh_tables (
                    table_id INTEGER PRIMARY KEY,
                    hyperplanes BLOB
                );
                """)

        cursor.execute("""
                CREATE TABLE lsh_hash_keys (
                    book_id INTEGER PRIMARY KEY,
                    FOREIGN KEY (book_id) REFERENCES books(book_id)
                );
                """)

        for table_id, table_id_name in enumerate(self.table_id_names):
            b_hyperplanes = np.random.randn(self.num_planes, embedding_dimension).astype(self.plane_type)
            cursor.execute("INSERT INTO lsh_tables (table_id, hyperplanes) VALUES (?, ?);", (table_id, b_hyperplanes))

            cursor.execute(f"""
                    ALTER TABLE lsh_hash_keys ADD COLUMN {table_id_name} BLOB;
                    """)

        cursor.connection.commit()

    def _hash(self, vector, hyperplanes):
        projections = vector @ hyperplanes.T
        return np.packbits(projections > 0)
    
    def get_hash_keys(self, cursor, vector):
        """Returns hash key for each table used for LSH. Assumes that vector is one dimension."""
        cursor.execute("""
                SELECT table_id, hyperplanes FROM lsh_tables;
                """)

        hash_keys = [b"" for _ in range(self.num_tables)]
        for table_id, b_hyperplanes in cursor.fetchall():
            hyperplanes = np.frombuffer(b_hyperplanes, dtype=self.plane_type).reshape((self.num_planes, vector.shape[0]))
            hash_keys[table_id] = self._hash(vector, hyperplanes)

        return hash_keys


    def update(self, cursor: sqlite3.Cursor, book_id: int, vector: np.ndarray):
        """
        Updates LSH tables with input vector.
        Assumes that the vector has NOT already been added.
        """

        hash_keys = tuple(self.get_hash_keys(cursor, vector))

        insertion_query = f"INSERT INTO lsh_hash_keys {('book_id',) + self.table_id_names}"
        insertion_query += f" VALUES ({book_id}, " + ', '.join('?' * self.num_tables) + ')'

        cursor.execute(insertion_query, hash_keys)
        cursor.connection.commit()

    def is_lsh_initialized(self, cursor: sqlite3.Cursor):
        cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name='lsh_tables';
        """)

        lsh_tables_initialized = bool(cursor.fetchone())

        cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name='lsh_hash_keys';
        """)

        lsh_hash_keys_table_initialized = bool(cursor.fetchone())

        return lsh_tables_initialized and lsh_hash_keys_table_initialized
