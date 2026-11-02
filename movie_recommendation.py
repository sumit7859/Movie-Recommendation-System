
# ==========================================
# MOVIE RECOMMENDATION SYSTEM
# Content-Based Filtering
# Using Python, Pandas, NLTK and Scikit-learn
# ==========================================


# 1. Import required libraries

import numpy as np
import pandas as pd
import ast
import pickle

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 2. Load datasets

credits_df = pd.read_csv('credits.csv')
movies_df = pd.read_csv('movies.csv')


# 3. Display complete dataset settings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


print("\n========== CREDITS DATASET ==========")
print(credits_df.head())

print("\n========== MOVIES DATASET ==========")
print(movies_df.head())


# 4. Merge both datasets

movies_df = movies_df.merge(credits_df, on='title')

print("\nMerged Dataset Shape:")
print(movies_df.shape)

print("\nMerged Dataset Information:")
movies_df.info()


# 5. Select required columns

movies_df = movies_df[
    [
        'movie_id',
        'title',
        'overview',
        'genres',
        'keywords',
        'cast',
        'crew'
    ]
]


print("\nSelected Columns:")
print(movies_df.head())


# 6. Check missing values

print("\nMissing Values:")
print(movies_df.isnull().sum())


# 7. Handle missing values

movies_df.dropna(inplace=True)

movies_df.reset_index(drop=True, inplace=True)

print("\nMissing Values After Cleaning:")
print(movies_df.isnull().sum())


# 8. Convert genres into a list

def convert(obj):

    L = []

    for i in ast.literal_eval(obj):
        L.append(i['name'])

    return L


movies_df['genres'] = movies_df['genres'].apply(convert)


# 9. Convert keywords into a list

movies_df['keywords'] = movies_df['keywords'].apply(convert)


# 10. Extract top 3 actors from cast

def convert3(obj):

    L = []

    counter = 0

    for i in ast.literal_eval(obj):

        if counter != 3:

            L.append(i['name'])

            counter += 1

        else:
            break

    return L


movies_df['cast'] = movies_df['cast'].apply(convert3)


# 11. Extract director from crew

def fetch_director(obj):

    L = []

    for i in ast.literal_eval(obj):

        if i['job'] == 'Director':

            L.append(i['name'])

            break

    return L


movies_df['crew'] = movies_df['crew'].apply(fetch_director)


# 12. Convert overview into a list

movies_df['overview'] = movies_df['overview'].apply(
    lambda x: x.split()
)


# 13. Remove spaces between names

movies_df['genres'] = movies_df['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies_df['keywords'] = movies_df['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies_df['cast'] = movies_df['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies_df['crew'] = movies_df['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)


# 14. Create tags column

movies_df['tags'] = (
    movies_df['overview'] +
    movies_df['genres'] +
    movies_df['keywords'] +
    movies_df['cast'] +
    movies_df['crew']
)


# 15. Create new dataframe

new_df = movies_df[['movie_id', 'title', 'tags']].copy()

print("\nNew DataFrame:")
print(new_df.head())


# 16. Convert list into string

new_df['tags'] = new_df['tags'].apply(
    lambda x: " ".join(x)
)


# 17. Convert tags into lowercase

new_df['tags'] = new_df['tags'].apply(
    lambda x: x.lower()
)


# 18. Apply stemming using PorterStemmer

ps = PorterStemmer()


def stem(text):

    y = []

    for i in text.split():

        y.append(ps.stem(i))

    return " ".join(y)


new_df['tags'] = new_df['tags'].apply(stem)


print("\nAfter Stemming:")
print(new_df.head())


# 19. Text Vectorization

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(new_df['tags'])


print("\nVector Shape:")
print(vectors.shape)


# 20. Calculate Cosine Similarity

similarity = cosine_similarity(vectors)


print("\nSimilarity Matrix Shape:")
print(similarity.shape)


# 21. Create Recommendation Function

def recommend(movie):

    movie = movie.strip()

    movie_matches = new_df[
        new_df['title'].str.lower() == movie.lower()
    ]

    if movie_matches.empty:

        print("\nMovie Not Found!")

        print("Please enter a valid movie name.")

        return

    movie_index = movie_matches.index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\n================================")
    print("Recommended Movies for:", movie)
    print("================================")

    for i in movies_list:

        print(new_df.iloc[i[0]].title)


# 22. Test Recommendation System

recommend('Avatar')

recommend('Iron Man')

recommend('Batman')


# 23. Save the Model

pickle.dump(
    new_df,
    open('movie_list.pkl', 'wb')
)

pickle.dump(
    similarity,
    open('similarity.pkl', 'wb')
)

print("\nModel Saved Successfully!")


# 24. Display Final DataFrame

print("\n========== FINAL DATAFRAME ==========")

print(new_df.head())


# 25. Display Final DataFrame Shape

print("\nFinal DataFrame Shape:")

print(new_df.shape)


# 26. Interactive Movie Recommendation

print("\n================================")
print("MOVIE RECOMMENDATION SYSTEM")
print("================================")

while True:

    movie_name = input(
        "\nEnter Movie Name (or type exit): "
    )

    if movie_name.lower() == 'exit':

        print("\nThank you for using Movie Recommendation System!")

        break

    recommend(movie_name)
