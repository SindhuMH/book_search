import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Load the FAISS index and metadata
index = faiss.read_index('books_index.faiss')
df = pd.read_csv('goodreads_data.csv')
df = df.dropna(subset=['Description'])
# Initialize the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

def search_books(query, top_k=10):
    # Generate query embedding
    query_embedding = model.encode([query])

    # Search the FAISS index
    distances, indices = index.search(query_embedding, 20)

    # Fetch results with scores and ratings
    results = []
    for i, idx in enumerate(indices[0]):
        result = df.iloc[idx]
        matching_score = 1 / (1 + distances[0][i])  # Convert distance to a score
        results.append({
            'title': result['Book'],
            'description': result['Description'][:150] + '...',  # 2 line description approximation
            'url': result['URL'],
            'author': result['Author'],
            'genre': result['Genres'],
            'rating': result['Avg_Rating'],
            'score': matching_score
        })

    # Sort results based on combined score (matching score and rating)
    results.sort(key=lambda x: (x['score'], x['rating']), reverse=True)

    # Select top k results
    top_results = results[:top_k]

    return top_results

# Streamlit app interface
st.title("Book Search Engine")

query = st.text_input("Enter a search query:")
if query:
    results = search_books(query)
    for result in results:
        st.markdown(f"### [{result['title']}]({result['url']})")
        st.markdown(f"**Author:** {result['author']}")
        st.markdown(f"**Genre:** {result['genre']}")
        st.markdown(f"{result['description']}")
