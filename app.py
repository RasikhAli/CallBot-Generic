from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import os

app = Flask(__name__)

# Path for saving and loading data
DATA_FOLDER = 'data/'
MODEL_PATH = 'LLM/paraphrase-MiniLM-L6-v2'

# Load model (shared across routes)
model = SentenceTransformer(MODEL_PATH)
index = None  # Initialize index

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    return text

# Function to retrieve similar queries
def retrieve_similar_queries(query, model, index, df, k=5):
    query_embedding = model.encode([clean_text(query)])
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i in range(k):
        query_text = df['Query'].iloc[indices[0][i]]
        response_text = df['Response'].iloc[indices[0][i]]
        context_text = df['Context'].iloc[indices[0][i]] if 'Context' in df.columns else 'N/A'
        
        results.append({
            'query': query_text,
            'response': response_text,
            'context': context_text,
            'distance': float(distances[0][i])
        })
    return results

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to download template
@app.route('/download_template', methods=['GET'])
def download_template():
    template_path = os.path.join(DATA_FOLDER, 'template.csv')
    if not os.path.exists(template_path):
        template_df = pd.DataFrame(columns=["ID", "Query", "Response", "Context"])
        template_df.to_csv(template_path, index=False)
    return send_file(template_path, as_attachment=True)

# Route to handle file upload and training
@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global index
    
    # Get the uploaded file
    file = request.files['file']

    # Get all form data
    selected_options = {}
    
    # Loop through all column select inputs and extract index
    for key in request.form:
        if key.startswith('column_type_'):
            column_index = key.split('_')[-1]  # Extract the index from the key (e.g., column_type_1 -> 1)
            selected_options[int(column_index)] = request.form[key]

    # Filter and print only "query" and "response" columns
    query_columns = []
    question_column = 0
    response_column = 1
    for index, value in selected_options.items():
        if value == 'question':  # Column marked as Question/Query
            question_column = index
            query_columns.append(f"Column at index {index} is 'query'")
        elif value == 'answer':  # Column marked as Answer/Response
            response_column = index
            query_columns.append(f"Column at index {index} is 'response'")
    
    # Print the query/response column indices
    # print("Query/Response Columns:", query_columns)
    

    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        # Save the uploaded file with a fixed name: "query_dataset"
        if file.filename.endswith('.csv'):
            file_path = os.path.join(DATA_FOLDER, 'query_dataset.csv')
            file.save(file_path)  # Save CSV directly
        else:
            # Save the uploaded Excel file temporarily
            file_path = os.path.join(DATA_FOLDER, 'query_dataset.xlsx')
            file.save(file_path)

            # Convert Excel to CSV
            df = pd.read_excel(file_path)
            csv_file_path = os.path.join(DATA_FOLDER, 'query_dataset.csv')
            df.to_csv(csv_file_path, index=False)  # Save as CSV

        # Read the dataset (from the saved CSV file)
        df = pd.read_csv(os.path.join(DATA_FOLDER, 'query_dataset.csv'))

        # Clean the queries and responses
        df['Query_cleaned'] = df.iloc[:, question_column].apply(clean_text)
        df['Response_cleaned'] = df.iloc[:, response_column].apply(clean_text)

        # Generate embeddings
        query_embeddings = model.encode(df['Query_cleaned'].tolist())
        d = query_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)  # Using Euclidean distance
        index.add(query_embeddings)

        # Save embeddings and index
        np.save(os.path.join(DATA_FOLDER, 'query_embeddings.npy'), query_embeddings)
        faiss.write_index(index, os.path.join(DATA_FOLDER, 'faiss_index.index'))

        # Redirect to the home route after success
        return redirect(url_for('home'))

    return jsonify({"error": "Invalid file format. Please upload a CSV or Excel file."}), 400

# Route to get similar queries from chatbot
@app.route('/get_similar_queries', methods=['POST'])
def get_similar_queries():
    query = request.form['query'].strip().lower()
    if not index:
        return jsonify([{'response': "Model not trained yet. Please upload a dataset first."}])

    # Load the trained dataset for response retrieval
    df = pd.read_csv(os.path.join(DATA_FOLDER, 'query_dataset.csv'))
    
    # Retrieve similar queries
    results = retrieve_similar_queries(query, model, index, df, k=1)
    return jsonify(results)

if __name__ == '__main__':
    # Ensure the data folder exists
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    app.run(debug=True)
