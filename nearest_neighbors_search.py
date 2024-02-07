import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# Sample DataFrame
data = {
    'id': [1, 2, 3, 4, 5],
    'url': ['url1', 'url2', 'url3', 'url4', 'url5'],
    'title': ['title1', 'title2', 'title3', 'title4', 'title5'],
    'desc': ['description1 fatigue', 'description2', 'fatigue description3', 'description4', 'no fatigue'],
    'location': ['loc1', 'loc2', 'loc3', 'loc4', 'loc5']
}

df = pd.DataFrame(data)

# Combine relevant text columns into a single column for vectorization
df['text'] = df['title'] + ' ' + df['desc']

# # Define a range of hyperparameters
# max_df_options = [0.75, 0.85, 1.0]  # Increased upper range
# min_df_options = [1, 2]  # Lower range
# ngram_range_options = [(1, 1), (1, 2)]
# n_neighbors_options = [2, 3, 4, 5]
#
# # Define a function to evaluate the model
# def evaluate_model(indices, relevant_documents):
#     # Count how many relevant documents are in the top N neighbors
#     count_relevant = sum([1 for idx in indices[0] if df.iloc[idx]['id'] in relevant_documents])
#     return count_relevant
#
# # Iterate over all combinations of hyperparameters
# best_score = 0
# best_params = {}
#
# for max_df in max_df_options:
#     for min_df in min_df_options:
#         for ngram_range in ngram_range_options:
#             for n_neighbors in n_neighbors_options:
#                 # Set up TF-IDF vectorization
#                 vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range)
#                 tfidf_matrix = vectorizer.fit_transform(df['text'])
#
#                 # Set up Nearest Neighbors model
#                 nn_model = NearestNeighbors(n_neighbors=n_neighbors)
#                 nn_model.fit(tfidf_matrix)
#
#                 # Transform the keyword and find neighbors
#                 keyword_vector = vectorizer.transform(['fatigue'])
#                 distances, indices = nn_model.kneighbors(keyword_vector)
#
#                 # Evaluate the model
#                 score = evaluate_model(indices, relevant_documents=[1, 2])  # Example relevant document IDs
#                 params = {'score': score, 'max_df': max_df, 'min_df': min_df, 'ngram_range': ngram_range, 'n_neighbors': n_neighbors}
#                 print(params)
#                 if score > best_score:
#                     best_score = score
#                     best_params = {'score': score, 'max_df': max_df, 'min_df': min_df, 'ngram_range': ngram_range, 'n_neighbors': n_neighbors}
#
# # Print the best parameters
# print("Best Parameters:", best_params)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['text'])

# Perform Nearest Neighbors search
N = 3
nn_model = NearestNeighbors(n_neighbors=N, algorithm='auto')
nn_model.fit(tfidf_matrix)

# Transform the keyword "fatigue" using the same TF-IDF vectorizer
keyword_vector = vectorizer.transform(['fatigue'])

# Find nearest neighbors for the keyword vector
distances, indices = nn_model.kneighbors(keyword_vector)

# Display the results
for i, index in enumerate(indices.flatten()):
    similarity_score = 1 - distances.flatten()[i]  # Convert cosine distance to similarity/
    print(
        f"ID: {df['id'][index]}, URL: {df['url'][index]}, Title: {df['title'][index]}, Desc: {df['desc'][index]}, Location: {df['location'][index]}")
print("Done...")
