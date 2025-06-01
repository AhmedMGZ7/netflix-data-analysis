# 🎬 Netflix Show Recommender

This project is a content-based recommendation system for Netflix titles using Python and Streamlit.

## Dataset
- https://www.kaggle.com/datasets/shivamb/netflix-shows
- download data from previous link before anything and if wanna use the python script should run the jupyter notebook

## 🚀 Features
- Input any Netflix movie or series title
- Get 5 similar titles based on content embeddings
- Simple and fast web UI using Streamlit

## 📊 Technologies Used
- Python
- pandas, numpy
- scikit-learn
- Streamlit

## 📷 Screenshots
![UI screenshot](images/ui.png)

## 🧠 How It Works
- Embeddings are extracted and stored
- Cosine similarity is used to compare titles
- User enters a title to get similar shows

## 🛠️ Run Locally
```bash
git clone ...
cd netflix-recommender
pip install -r requirements.txt
streamlit run app.py
