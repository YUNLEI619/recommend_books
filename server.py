import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

df = pd.read_csv('book_list.csv')

df.head(10)

# Filter out missing values (NaN) in 'Title', 'Author', and 'Publisher' columns
df_cleaned = df.dropna(subset=['Title', 'Author', 'Publisher']).copy()
df_cleaned.head()

# df_cleaned.drop(columns=['Unnamed: 6'], inplace=True)
# df_cleaned.head()


# Function to clean the author names
def clean_author_name(author_name):
    return author_name.split(',')[0]


# Apply the clean_author_name function to the 'Author' column
df_cleaned['Author'] = df_cleaned['Author'].apply(clean_author_name)

df_cleaned.head()


# Function to convert the Genre strings to lists of genres
def convert_genres(genre_str):
    try:
        genre_list = ast.literal_eval(genre_str)
        if not isinstance(genre_list, list):
            genre_list = [genre_list]
        return genre_list
    except (ValueError, SyntaxError):
        return []


# Convert the Genre strings to lists of genres
df['Genre'] = df['Genre'].apply(convert_genres)


def clean_isbn(isbn):
    isbn_digits = ''.join(filter(str.isdigit, isbn))
    if len(isbn_digits) == 13 and isbn_digits != '9999999999999' and isbn_digits != '0000000000000':
        return isbn_digits
    else:
        return None


# Apply the clean_isbn function to the 'ISBN' column
df_cleaned['ISBN'] = df_cleaned['ISBN'].astype(str).apply(clean_isbn)

# Drop rows with invalid ISBN (None)
df_cleaned.dropna(subset=['ISBN'], inplace=True)


def vectorize_text(text):
    words = text.split()
    vec = np.zeros(model.vector_size)
    count = 0
    for word in words:
        if word in model.wv:
            vec += np.array(model.wv[word])
            count += 1
    if count > 0:
        return vec / count
    return vec


# Create a Word2Vec model with cleaned data
df_cleaned['Text'] = df_cleaned['Title'] + " " + df_cleaned['Author'] + " " + df_cleaned['Genre'].apply(lambda x: ' '.join(x))
model = Word2Vec(sentences=df_cleaned['Text'].str.split(), vector_size=100, window=5, min_count=1, workers=4)

# Vectorize the Text column using .apply
df_cleaned.loc[:, 'TextVector'] = df_cleaned['Text'].apply(vectorize_text)

# Drop any rows containing NaN values in vectors
df_cleaned.dropna(subset=['TextVector'], inplace=True)

# Fill any remaining NaN values for 'TextVector' with zeros
df_cleaned['TextVector'].fillna(0, inplace=True)

# Create similarity matrices
title_sim_matrix = cosine_similarity(df_cleaned['TextVector'].tolist())
author_sim_matrix = cosine_similarity(df_cleaned['TextVector'].tolist())

# Combine the similarity matrices
combined_vectors = np.column_stack(df_cleaned['TextVector'].apply(lambda x: np.nan_to_num(x)).tolist())
combined_sim_matrix = cosine_similarity(combined_vectors)

rating_sim_matrix = 1 - np.abs(df_cleaned['Rating'].values[:, None] - df_cleaned['Rating'].values) / 5


def recommend_books(book_title, num_recommendations=5):
    if len(df_cleaned) == 0:
        return "Not enough books to provide recommendations."

    if book_title not in df_cleaned['Title'].values:
        return f"Book with title '{book_title}' not found in the database."

    target_index = df_cleaned[df_cleaned['Title'] == book_title].index[0]

    similar_books_indices = np.argsort(title_sim_matrix[target_index])[::-1]

    recommended_books = df_cleaned.iloc[similar_books_indices[1:num_recommendations + 1]]

    # Select only the 'Title', 'Author', and 'ISBN' columns
    recommended_books = recommended_books[['Title', 'Author', 'ISBN']]

    return recommended_books


# # Check for any remaining NaN values
# print(df_cleaned.isnull().sum())
#
# print(df_cleaned.isnull().sum())
# # Check the number of books in the dataset
# num_books = len(df_cleaned)
# print("Number of books in the dataset:", num_books)
#
# recommendations = recommend_books(input("Please input your favorite book: "), num_recommendations=5)
# print(recommendations)

# Replace the title of your choice for book recommendation


@app.route('/api/recommend', methods=['GET'])
def recommend_api():
    book_title = request.args.get('book_title')
    num_recommendations = int(request.args.get('num_recommendations', 5))

    recommendations = recommend_books(book_title, num_recommendations=num_recommendations)

    recommendations_list = recommendations.to_dict(orient='records')

    return jsonify(recommendations_list)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/api/mock', methods=['GET'])
def mock():
    return '123'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
