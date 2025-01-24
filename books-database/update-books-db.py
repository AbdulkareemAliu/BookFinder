import csv
import json
import sqlite3
import argparse
import urllib.parse
import urllib.request
from collections.abc import Callable
from sentence_transformers import SentenceTransformer

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
        compute_embeddings: bool=False, embedding_model: SentenceTransformer = None
    ) -> None:
    assert not compute_embeddings or embedding_model is not None, "If you would like to use embeddings, make sure to provide embedding model"

    with open(filepath, "r") as csv_file:
        reader = csv.reader(csv_file)
        for entry in reader:
            assert len(entry) >= 3, f'Need at least 3 elements per row, found: {len(entry)} elements: {entry}'
            title, authors, shelf_row = entry[0].strip().lower(), entry[1].strip().lower(), entry[2].strip().lower()
            if does_book_exist(cursor, title, authors):
                continue
            
            if generate_description:
                description = get_description(title, authors)
            else:
                description = entry[3].strip() if len(entry) > 3 else 'No description found' 

            print(f"Adding title: {title} ~ authors: {authors} ~ shelf row: {shelf_row} ~ description: {description[:35]}")
            title_embedding = embedding_model.encode(title + "; author: " + authors).tobytes() if compute_embeddings else None
            description_embedding = embedding_model.encode(description).tobytes() if compute_embeddings and description else None
            insert_query = """INSERT OR IGNORE
                INTO books(
                    \"title\",
                    \"authors\",
                    \"shelf_row\",
                    \"description\",
                    \"title_embedding\",
                    \"description_embedding\"
                ) VALUES(?, ?, ?, ?, ?, ?);
            """
            cursor.execute(insert_query, (title, authors, shelf_row, description, title_embedding, description_embedding))
            cursor.connection.commit()
            
def initialize_fts(cursor: sqlite3.Cursor) -> None:
    cursor.executescript('''
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
    cursor.connection.commit()

if __name__ == '__main__':
    books_db = sqlite3.connect("books.db")
    cursor = books_db.cursor()

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        authors TEXT NOT NULL,
        shelf_row INTEGER NOT NULL,
        description TEXT,
        title_embedding BLOB,
        description_embedding BLOB,
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
    parser.add_argument('--initialize_fts', action='store_true')
    parser.add_argument('--library_api', choices=['google_books', 'open_library'], default='google_books')
    args = parser.parse_args()

    arg_to_get_description = {
        "google_books": get_description_google_books,
        "open_library": get_description_open_library
    }

    update_books(cursor, args.filepath, args.generate_description, arg_to_get_description[args.library_api], args.compute_embeddings, embedding_model)

    if (args.initialize_fts):
        initialize_fts(cursor)

    books_db.close()
