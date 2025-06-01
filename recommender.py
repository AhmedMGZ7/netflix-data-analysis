import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"File {filepath} not found. Please check the path.")
        return None

def recommend(title, df, top_n=5):
    if title not in df['title'].values:
        print(f"Sorry, '{title}' not found in the dataset.")
        return []

    query_vec = df.loc[df['title'] == title, 'embedding'].values[0]
    similarities = cosine_similarity([query_vec], df['embedding'].tolist())[0]
    sorted_indices = similarities.argsort()[::-1][1:top_n+1]
    results = [(df.iloc[i]['title'], similarities[i]) for i in sorted_indices]
    return results

def main():
    data_path = 'cleaned_netflix_data.csv'
    df = load_data(data_path)
    # Convert the string embeddings back to numpy arrays
    df['embedding'] = df['embedding'].apply(
    lambda x: np.array([float(i) for i in x.strip('[]').split()])
    )
    if df is None:
        return  # exit if data not found

    print("Welcome to the Netflix Show Recommender!")
    print("Type 'exit' anytime to quit.\n")

    while True:
        title = input("Enter a movie or show title: ").strip()
        if title.lower() == 'exit':
            print("Goodbye!")
            break

        recommendations = recommend(title, df)
        if recommendations:
            print(f"\nRecommendations similar to '{title}':")
            for rec_title, score in recommendations:
                print(f"- {rec_title} (Similarity: {score:.3f})")
        print()

if __name__ == "__main__":
    main()
