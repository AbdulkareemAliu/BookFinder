import argparse
from speech_to_text import SpeechHandler
from book_search import BookSearchHandler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', choices=[
        'fts', 'embedding_naive', 'embedding_cluster', 'embedding_lsh', 'embedding_cluster_lsh'
    ], default='embedding_naive')
    parser.add_argument('--similarity_threshold', default=0.3)
    parser.add_argument('--quantize_embedding', action='store_true')
    args = parser.parse_args()

    speech_handler = SpeechHandler("../models/vosk-model-small-en-us-0.15")
    search_handler = BookSearchHandler(
        search_method=args.method,
        quantize_embeddings=args.quantize_embedding,
        similarity_threshold=float(args.similarity_threshold)
    )
    num_queries = 0
    while num_queries < 3:
        query = speech_handler.record(" ")
        print(f"You asked: {query}")
        if not query:
            continue
        num_queries += 1
        result = search_handler.book_search(query)

        for r in result:
            print(r)