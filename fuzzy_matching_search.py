import pandas as pd
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
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

# Combine relevant text columns into a single column for matching
df['text'] = df['title'] + ' ' + df['desc']

# Find the closest matches to the keyword "fatigue"
N = 5
matches = process.extract("fatigue", df['text'], limit=N)

# Display the results
for match, score, index in matches:
    print(f"Score: {score}, ID: {df['id'][index]}, URL: {df['url'][index]}, Title: {df['title'][index]}, Desc: {df['desc'][index]}, Location: {df['location'][index]}")
    print("-------------")
