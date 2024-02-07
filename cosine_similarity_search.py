import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV

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

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['text'])

# Transform the keyword "fatigue" using the same TF-IDF vectorizer
keyword_vector = vectorizer.transform(['fattigue'])

# Calculate cosine similarity
cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix).flatten()

# Get indices of top N similar items
N = 3
top_indices = cosine_similarities.argsort()[-N:][::-1]

# Display the results
for index in top_indices:
    print(
        f"ID: {df['id'][index]}, URL: {df['url'][index]}, Title: {df['title'][index]}, Desc: {df['desc'][index]}, Location: {df['location'][index]}")
print("Done...")
