import json
import csv
import sqlite3
import argparse
import urllib.parse
import urllib.request

def get_description_google_books(title, authors):
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
    
def get_description_open_library(title, authors):
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
    
def does_book_exist(cursor, title, authors):
    cursor.execute("""
        SELECT COUNT(1)
        FROM books
        WHERE (\"title\", \"authors\") = (?, ?);
    """, (title, authors))

    return bool(cursor.fetchone()[0])
    
def update_books(cursor, filepath, generate_description=False, get_description=get_description_google_books):
    assert not generate_description or get_description is not None, "Description not included in input and get description method not provided"

    with open(filepath, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader) #skipping header
        for entry in reader:
            assert len(entry) >= 3, f'Need at least 3 elements per row, found: {len(entry)} elements: {entry}'
            title, authors, shelf_row = entry[0], entry[1], entry[2]
            if does_book_exist(cursor, title, authors):
                print('skip')
                continue
            
            if generate_description:
                description = get_description(title, authors)
            else:
                description = entry[3] if len(entry) > 3 else 'No description found' 

            print(f"Adding title: {title} ~ authors: {authors} ~ shelf row: {shelf_row} ~ description: {description[:35]}")
            cursor.execute(f"INSERT OR IGNORE INTO books(\"title\", \"authors\", \"shelf_row\", \"book_description\") VALUES(?, ?, ?, ?);", (title, authors, shelf_row, description))
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
        book_description TEXT,
        UNIQUE(title, authors)
    );"""
    )
    parser = argparse.ArgumentParser(
                    prog='UpdateBooksDB',
                    description='Updates the books database used by the program given a text file of name, authors, and row number for books to be added. See ./sample_books_*_description.csv for format of this list and the books considered in this project. Can submit lists with and without descriptions. If a list without a description is used, a description will be fetched using the passed in web api (google books by default).')

    parser.add_argument('filepath')
    parser.add_argument('--generate_description', action='store_true')
    parser.add_argument('--library_api', choices=['google_books', 'open_library'], default='google_books')
    args = parser.parse_args()

    arg_to_get_description = {
        "google_books": get_description_google_books,
        "open_library": get_description_open_library
    }

    update_books(cursor, args.filepath, args.generate_description, arg_to_get_description[args.library_api])
