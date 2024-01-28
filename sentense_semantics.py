# Documentation: https://www.sbert.net/examples/applications/semantic-search/README.html
from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
# corpus_embeddings = corpus_embeddings.to('cuda')
# corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

# Query sentences:
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.',
           'A cheetah chases prey on across a field.']

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embeddings = embedder.encode(query, convert_to_tensor=True)
    # query_embeddings = query_embeddings.to('cuda')
    # query_embeddings = util.normalize_embeddings(query_embeddings)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    # cos_scores = util.cos_sim(query_embeddings, corpus_embeddings)[0]
    # top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    # for score, idx in zip(top_results[0], top_results[1]):
    #     print(corpus[idx], "(Score: {:.4f})".format(score))

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    """
    # hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=5)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
