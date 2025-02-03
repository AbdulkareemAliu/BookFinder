# BookFinder
Implemented vector search for books in a bookshelf based on their titles and descriptions. The title, authors, shelf position, description, and the embedding of each book is stored in an sqlite database. Also finetuned the snowflake-arctic-embed-m-v1.5 model to be more compatible with book data, and the relevant code is featured in main/finetune_embedding_model notebook. I intended for this system to be lightweight and able to be run on a raspberry pi.

### main/update_books_db.csv 
Add to the database from a csv. If you do not have access to the description, you can pass in the --generate_description flag and the google books or open library api will be called to find one. You can pass in a variety of other flags to control which features of the system you would like to use, including and not limited to, embedding quantization, fts support, or k means clustering.

### Search Methods
Investigated FTS search and embedding search. Since the system supports vocal queries, fts, although much faster, often does not return any meaningful results, and embedding search is much more suited for natural language comparatively. With embedding similarity search, however, additional methods needed to be investigated to limit the search space before performing the cosine similarity operation. The two methods used in that effort were K-Means clustering (main/cluster_embeddings.py) and locality sensitive hashing (main/lsh_implementation.py). 

With K Means, some constant number of centroids are precomputed and stored in a table in the database. When a query is made, it is first compared to these centroids, and then against all other books with a matching centroid ID. 
With locality sensitive hashing, we generate some fixed number (t) of tables that contain of another fixed number (h) of vectors that define hyperplanes. Each embedding vector is assigned a hash value that corresponds to its dot product with the hyperplanes of the table. During search, we only search through puts with at least 1 matching hash value.

These methods can also be combined to further reduce the search space. This is done by only searching through books within the query centroid and that have a matching hash value.

### Finetuning
The embedding model was finetuned using this dataset https://www.kaggle.com/datasets/elvinrustam/books-dataset and randomly generated queries (details in main/finetune_embedding_model)

### Query system
main/book_search.py is a script that allows you to make queries through the terminal and main/main.py allows you to make queries with speech. Both systems allow you to specify the method you would like to make searches, the minimum similarity threshold needed in your results, and the maximum number of results to return.