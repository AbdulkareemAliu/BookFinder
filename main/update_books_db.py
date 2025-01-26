import csv
import json
import sqlite3
import argparse
import numpy as np
import urllib.parse
import urllib.request
from lsh_implementation import LSH
from collections.abc import Callable
from sentence_transformers import SentenceTransformer
from cluster_embeddings import are_books_clustered, find_nearest_centroid, cluster

def get_description_google_books(title: str, authors: str) -> str:
    encoded_title = urllib.parse.quote_plus(title)
    encoded_authors = urllib.parse.quote_plus(authors)
    url = f"https://www.googleapis.com/books/v1/volumes?q=title:{encoded_title}+inauthor:{encoded_authors}"
    
    try:
        with urllib.request.urlopen(url) as f:
            json_object = json.load(f)
            if (
                json_object['totalItems'] > 0 and json_object['items'] and 
                'volumeInfo' in json_object['items'][0] and 'description' in json_object['items'][0]['volumeInfo']
            ):
                return json_object['items'][0]['volumeInfo']['description']
            else:
                return "No description found"
            
        assert False, f"Failed to make request for book - title: {title} and authors: {authors}"
    except urllib.error.URLError as e:
        print(f"Error fetching data from Google Books: {e}")
        return "No description found"
    
def get_description_open_library(title: str, authors: str) -> str:
    encoded_title = urllib.parse.quote_plus(title)
    encoded_authors = urllib.parse.quote_plus(authors)
    url = f"https://openlibrary.org/search.json?q=title:{encoded_title}+author:{encoded_authors}"
    
    try:
        with urllib.request.urlopen(url) as f:
            json_object = json.load(f)
            if ('docs' in json_object and json_object['docs'] and 'description' in json_object['docs'][0]):
                return json_object['items'][0]['volumeInfo']['description']
            else:
                return "No description found"
            
        assert False, f"Failed to make request for book - title: {title} and authors: {authors}"
    except urllib.error.URLError as e:
        print(f"Error fetching data from Open Library: {e}")
        return "No description found"
    
def does_book_exist(cursor: sqlite3.Cursor, title: str, authors: str) -> bool:
    cursor.execute("""
        SELECT COUNT(1)
        FROM books
        WHERE (\"title\", \"authors\") = (?, ?);
    """, (title, authors))

    return bool(cursor.fetchone()[0])
    
def update_books(
        cursor: sqlite3.Cursor,
        filepath: str, 
        generate_description: bool=False, 
        get_description: Callable[[str, str], str] = get_description_google_books, 
        compute_embeddings: bool=False, 
        embedding_model: SentenceTransformer = None,
        recluster_embeddings: bool=False,
        use_lsh: bool = False,
        lsh: LSH = None
    ) -> None:
    assert not compute_embeddings or embedding_model is not None, "If you would like to use embeddings, make sure to provide embedding model"
    assert not use_lsh or lsh is not None, "Must bass in lsh object if you would like to update with LSH"

    with open(filepath, "r") as csv_file:
        reader = csv.reader(csv_file)
        for book_id, entry in enumerate(reader):
            assert len(entry) >= 3, f'Need at least 3 elements per row, found: {len(entry)} elements: {entry}'
            title, authors, shelf_row = entry[0].strip().lower(), entry[1].strip().lower(), entry[2].strip().lower()
            if does_book_exist(cursor, title, authors):
                continue
            
            if generate_description:
                description = get_description(title, authors)
            else:
                description = entry[3].strip() if len(entry) > 3 else 'No description found' 

            print(f"Adding title: {title} ~ authors: {authors} ~ shelf row: {shelf_row} ~ description: {description[:35]}")
            centroid_id = -1
            if compute_embeddings:
                embedding = embedding_model.encode(title + "; author: " + authors, normalize_embeddings=True) + embedding_model.encode(description, normalize_embeddings=True)

                # will only assign embedding cluster id if there exists a cluster table and we do not plan to recluster
                if not recluster_embeddings and are_books_clustered(cursor):
                    centroid_id = find_nearest_centroid(cursor, embedding)

                if use_lsh and lsh.is_lsh_initialized(cursor):
                    lsh.update(cursor, book_id, embedding)

                embedding = embedding.tobytes()

            else:
                embedding = None

            insert_query = """INSERT OR IGNORE
                INTO books(
                    \"book_id\",
                    \"title\",
                    \"authors\",
                    \"shelf_row\",
                    \"description\",
                    \"embedding\",
                    centroid_id
                ) VALUES(?, ?, ?, ?, ?, ?, ?);
            """

            cursor.execute(insert_query, (book_id, title, authors, shelf_row, description, embedding, centroid_id))
            cursor.connection.commit()
            
def initialize_fts(cursor: sqlite3.Cursor) -> None:
    cursor.executescript('''
        CREATE VIRTUAL TABLE books_fts USING fts5(
            title, authors, description
        );
                                
        INSERT INTO books_fts (rowid, title, authors, description)
        SELECT book_id, title, authors, description FROM books;

        CREATE TRIGGER books_ai AFTER INSERT ON books BEGIN
            INSERT INTO books_fts(rowid, title, authors, description)
            VALUES (new.book_id, new.title, new.authors, new.description);
        END;

        CREATE TRIGGER books_ad AFTER DELETE ON books BEGIN
            DELETE FROM books_fts
            WHERE rowid = old.book_id;
        END;

        CREATE TRIGGER books_au AFTER UPDATE ON books BEGIN
            UPDATE books_fts
            SET 
                title = new.title,
                authors = new.authors,
                description = new.description
            WHERE rowid = new.book_id;
        END;
    ''')
    cursor.connection.commit()

if __name__ == '__main__':
    books_db = sqlite3.connect("../books-database/books.db")
    cursor = books_db.cursor()

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        authors TEXT NOT NULL,
        shelf_row INTEGER NOT NULL,
        description TEXT,
        embedding BLOB,
        centroid_id INTEGER,
        UNIQUE(title, authors)
    );"""
    )

    # Run this if you do not have the model saved locally
    # embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")
    # embedding_model.save("../models/snowflake-arctic-embed-m-v1.5")

    embedding_model = SentenceTransformer("../models/finetuned-snowflake-arctic-embed-m-v1.5")

    parser = argparse.ArgumentParser(
                    prog='UpdateBooksDB',
            )

    parser.add_argument('filepath')
    parser.add_argument('--generate_description', action='store_true')
    parser.add_argument('--compute_embeddings', action='store_true')
    parser.add_argument('--quantize_embeddings', action='store_true')
    parser.add_argument('--recluster_embeddings', action='store_true')
    parser.add_argument('--use_lsh', action='store_true')
    parser.add_argument('--initialize_fts', action='store_true')
    parser.add_argument('--library_api', choices=['google_books', 'open_library'], default='google_books')
    args = parser.parse_args()

    arg_to_get_description = {
        "google_books": get_description_google_books,
        "open_library": get_description_open_library
    }

    lsh = LSH()

    if (args.use_lsh and not lsh.is_lsh_initialized(cursor)):
        lsh.reset_tables(cursor, embedding_model.get_sentence_embedding_dimension())

    update_books(
        cursor, 
        args.filepath, 
        args.generate_description, 
        arg_to_get_description[args.library_api], 
        args.compute_embeddings, 
        embedding_model,
        args.recluster_embeddings,
        args.use_lsh,
        lsh
    )

    if (args.initialize_fts):
        initialize_fts(cursor)

    if (args.recluster_embeddings):
        cluster(cursor, 5)

    books_db.close()
