import sqlite3
from typing import List, Tuple
from collections import Counter

class BookSearch:
    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        
    def setup_fts(self) -> None:
        self.cursor.executescript('''
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
        self.connection.commit()

    def fts_search(self, query: str) -> List[Tuple]:
        sql = '''
        SELECT books.id, books.title, books.authors, books.shelf_row, books_fts.rank
        FROM books
        INNER JOIN books_fts ON books.id = books_fts.rowid
        WHERE books_fts.description MATCH ?
        ORDER BY rank
        '''
        return self.cursor.execute(sql, (query,)).fetchall()

    def close(self):
        self.connection.close()

if __name__ == '__main__':
    searcher = BookSearch("../books-database/books.db")
    query = "economics"
    print(searcher.fts_search(query))
    searcher.close()