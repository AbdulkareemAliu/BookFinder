import argparse
from speech_to_text import SpeechHandler
from book_search import BookSearchHandler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', choices=[
        'fts', 'embedding_naive', 'embedding_cluster', 'embedding_lsh', 'embedding_cluster_lsh'
    ], default='embedding_naive')
    parser.add_argument('--similarity_threshold', default=0.3)
    parser.add_argument('--max_num_results', default=5)
    parser.add_argument('--max_queries', default=10)
    parser.add_argument('--key', default=" ")
    parser.add_argument('--quantize_embedding', action='store_true')
    args = parser.parse_args()

    speech_handler = SpeechHandler("../models/vosk-model-small-en-us-0.15")
    search_handler = BookSearchHandler(
        search_method=args.method,
        quantize_embeddings=args.quantize_embedding,
        similarity_threshold=float(args.similarity_threshold),
        max_num_results=int(args.max_num_results)
    )
    start_key = "spacebar" if args.key == " " else args.key

    print(f"______________________________________________________________")
    print(f"Hello. Press the {start_key} key to start recording you query.")
    print(f"Please wait until what you have said is printed to the console before submitting the query.")
    print(f"Press the {start_key} key again to submit your query for search. Say \'stop\' to stop")
    print(f"You can make a maximum of {args.max_queries} queries. You can change this number with --max_queries flag and change the start/stop key with the --key flag.")
    num_queries = 0
    while num_queries < int(args.max_queries):
        query = speech_handler.record(args.key)
        print(f"You asked: {query}")
        if not query:
            continue
        if query.strip().lower() == "stop":
            break
        num_queries += 1
        result = search_handler.book_search(query)
        print("Found: ")
        for r in result:
            print(r)