import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re

# Load dummy dataset (replace with actual admissions queries dataset)
df = pd.read_csv('data/admission_queries_responses4.csv')

# Data cleaning function (optional based on input data)
def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower().strip()
    return text

# Apply cleaning if necessary
df['Query_cleaned'] = df['Query'].apply(clean_text)
df['Response_cleaned'] = df['Response'].apply(clean_text)

# Initialize model and compute embeddings for the queries
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model_path_en = 'LLM/paraphrase-MiniLM-L6-v2'
model = SentenceTransformer(model_path_en)

query_embeddings = model.encode(df['Query_cleaned'].tolist())

# Create FAISS index
d = query_embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)  # Using Euclidean distance for similarity search
index.add(query_embeddings)  # Add embeddings to index

# Save embeddings and index for future use
np.save('data/admission_query_embeddings.npy', query_embeddings)
faiss.write_index(index, 'data/admission_faiss.index')

# Function to retrieve similar queries
def retrieve_similar_queries(query, model, index, df, k=1):
    query_embedding = model.encode([clean_text(query)])
    distances, indices = index.search(query_embedding, k)

    for i in range(k):
        print(f"Query {i+1}:")
        print(df['Query'].iloc[indices[0][i]])
        print(f"Response: {df['Response'].iloc[indices[0][i]]}")
        print(f"Distance: {distances[0][i]}\n")


index = faiss.read_index('data/admission_faiss.index')
query = "What are the admission requirements?"
retrieve_similar_queries(query, model, index, df)

query = "Can I get a scholarship?"
retrieve_similar_queries(query, model, index, df)
