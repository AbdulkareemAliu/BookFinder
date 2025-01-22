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
        print(f"Error fetching data from Open Library: {e}")
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
        compute_embedding: bool=False, embedding_model: SentenceTransformer = None
    ) -> None:
    assert not compute_embedding or embedding_model is not None, "If you would like to use embeddings, make sure to provide embedding model"

    with open(filepath, "r") as csv_file:
        reader = csv.reader(csv_file)
        for entry in reader:
            assert len(entry) >= 3, f'Need at least 3 elements per row, found: {len(entry)} elements: {entry}'
            title, authors, shelf_row = entry[0].strip(), entry[1].strip(), entry[2].strip()
            if does_book_exist(cursor, title, authors):
                print('skip')
                continue
            
            if generate_description:
                description = get_description(title, authors)
            else:
                description = entry[3].strip() if len(entry) > 3 else 'No description found' 

            print(f"Adding title: {title} ~ authors: {authors} ~ shelf row: {shelf_row} ~ description: {description[:35]}")
            embedding = embedding_model.encode(description).tobytes() if compute_embedding else None
            cursor.execute(f"INSERT OR IGNORE INTO books(\"title\", \"authors\", \"shelf_row\", \"description\", \"embedding\") VALUES(?, ?, ?, ?, ?);", (title, authors, shelf_row, description, embedding))
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
        embedding BLOB,
        UNIQUE(title, authors)
    );"""
    )

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    parser = argparse.ArgumentParser(
                    prog='UpdateBooksDB',
            )

    parser.add_argument('filepath')
    parser.add_argument('--generate_description', action='store_true')
    parser.add_argument('--compute_description_embedding', action='store_true')
    parser.add_argument('--library_api', choices=['google_books', 'open_library'], default='google_books')
    args = parser.parse_args()

    arg_to_get_description = {
        "google_books": get_description_google_books,
        "open_library": get_description_open_library
    }

    update_books(cursor, args.filepath, args.generate_description, arg_to_get_description[args.library_api], args.compute_description_embedding, embedding_model)
    books_db.close()
