import csv
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

def save_calibration_embeddings(
        filepath: str, 
        cursor: sqlite3.Cursor, 
        embedding_model: SentenceTransformer,
        max_embeddings: int = 500
):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS calibration_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding BLOB
    );
    """)

    cursor.execute("""
        SELECT COUNT(*) FROM calibration_embeddings;
    """)
    num_embeddings = cursor.fetchone()[0]
    if num_embeddings >= max_embeddings:
        print("Too many embeddings saved in calibration dataset, will save no more")
        return
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calibration_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB
        );
        """)
    insert_query = "INSERT INTO calibration_embeddings (embedding) VALUES (?);"

    cursor.connection.commit()

    with open(filepath, "r") as csv_file:
        reader = csv.reader(csv_file)
        for _, entry in enumerate(reader):
            num_embeddings += 1
            if num_embeddings >= max_embeddings:
                print("Too many embeddings saved in calibration dataset, will save no more")
                return
            assert len(entry) >= 3, f'Need at least 3 elements per row, found: {len(entry)} elements: {entry}'
            title, authors, _ = entry[0].strip().lower(), entry[1].strip().lower(), entry[2].strip().lower()
            description = entry[3].strip().lower() if len(entry) > 3 and entry else ""

            b_embedding = embedding_model.encode(title + "; author: " + authors)
            if description:
                descr_embedding = embedding_model.encode(description)
                b_embedding = b_embedding + descr_embedding
            b_embedding = b_embedding.tobytes()
            cursor.execute(insert_query, (b_embedding,))

    cursor.connection.commit()

def get_calibration_embeddings(cursor: sqlite3.Cursor):
    cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name='calibration_embeddings';
        """)
    
    if not cursor.fetchone():
        print("calibration embeddings table does not exist")
        return []
    
    cursor.execute("SELECT embedding FROM calibration_embeddings")
    b_embeddings = cursor.fetchall()

    return [np.frombuffer(b_embedding[0], dtype=np.float32) for b_embedding in b_embeddings]

if __name__ == "__main__":
    books_db = sqlite3.connect("../books-database/books.db")
    cursor = books_db.cursor()

    embedding_model = SentenceTransformer("../models/finetuned-snowflake-arctic-embed-m-v1.5")

    filepath = "../books-database/sample_books_with_description.csv"

    save_calibration_embeddings(
        filepath,
        cursor,
        embedding_model,
        1000
    )

    print(len(get_calibration_embeddings(cursor)))