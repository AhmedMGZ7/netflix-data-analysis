import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load your data
df = pd.read_csv('cleaned_netflix_data.csv')  # Must have 'title' and 'embedding' columns
df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))

# Recommender function
def recommend(title, k=5):
    title = title.lower()
    if title not in df['title'].str.lower().values:
        return []
    
    idx = df[df['title'].str.lower() == title].index[0]
    sim_scores = cosine_similarity([df.loc[idx, 'embedding']], df['embedding'].tolist())[0]
    
    top_indices = np.argsort(sim_scores)[::-1][1:k+1]
    recommendations = [(df.iloc[i]['title'], round(sim_scores[i], 3)) for i in top_indices]
    return recommendations

# Streamlit UI
st.title("🎬 Netflix Show Recommender")
user_input = st.text_input("Enter a movie or show title:")

if user_input:
    recs = recommend(user_input)
    if recs:
        st.success(f"Recommendations similar to '{user_input}':")
        for title, score in recs:
            st.markdown(f"- **{title}** (Similarity: {score})")
    else:
        st.error("Title not found. Try another Netflix title.")
