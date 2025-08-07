import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
# CHANGE 1: Import Groq instead of OpenAI
import groq 
import os
import time

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = 'processed_customer_data.csv'
PRODUCT_KB_PATH = 'products.txt'
INDEX_PATH = 'faiss_index.bin' 

# CHANGE 2: Load the Groq API key from environment variables
try:
    # Note: It's good practice to use a specific client instance
    groq_client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
except KeyError:
    print("ERROR: GROQ_API_KEY environment variable not set.")
    print("Please set the key and restart.")
    exit()

def build_or_load_index():
    """
    Builds a FAISS index from the product knowledge base or loads it if it already exists.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if os.path.exists(INDEX_PATH):
        print(f"Loading existing FAISS index from '{INDEX_PATH}'...")
        index = faiss.read_index(INDEX_PATH)
        with open(PRODUCT_KB_PATH, 'r', encoding='utf-8') as f:
            product_texts = f.read().split('--------------------------------')
        return index, model, product_texts

    print("Building new FAISS index...")
    with open(PRODUCT_KB_PATH, 'r', encoding='utf-8') as f:
        product_texts = f.read().split('--------------------------------')
        product_texts = [text.strip() for text in product_texts if text.strip()]

    print(f"Creating embeddings for {len(product_texts)} product descriptions...")
    embeddings = model.encode(product_texts, convert_to_tensor=False)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    print(f"Saving index to '{INDEX_PATH}'...")
    faiss.write_index(index, INDEX_PATH)
    
    return index, model, product_texts

def create_customer_query(customer_data):
    """
    Creates a natural language query describing the customer's profile.
    """
    query = (
        f"Customer profile: "
        f"Age {customer_data['age']}. "
        f"Income of ${customer_data['income']:.0f}. "
        f"Credit score is {customer_data['credit_score']}. "
        f"Debt-to-income ratio is {customer_data['debt_to_income_ratio']:.2f}. "
        f"Holds {customer_data['product_count']} products, including: {customer_data['existing_products']}. "
        f"Digital engagement score is {customer_data['engagement_score']}. "
        f"Employment status: {customer_data['employment_status']}."
    )
    return query

def get_recommendation(customer_id, customer_df, index, model, product_texts):
    """
    Generates a product recommendation for a given customer_id using the RAG process.
    """
    print(f"\n--- Generating recommendation for Customer ID: {customer_id} ---")
    
    try:
        customer_data = customer_df.loc[customer_df['customer_id'] == customer_id].iloc[0]
    except IndexError:
        return "Error: Customer ID not found."

    query_text = create_customer_query(customer_data)
    print(f"Generated Search Query: \"{query_text}\"")
    
    query_vector = model.encode([query_text])
    k = 3 
    distances, indices = index.search(query_vector, k)
    retrieved_docs = [product_texts[i] for i in indices[0]]
    
    print(f"\nRetrieved Top {k} Relevant Product Docs:")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i+1}. {doc.splitlines()[0]}")

    prompt = f"""
    You are an expert banking relationship manager. Your task is to recommend the single best product for a customer based on their profile and a list of relevant products.

    **Customer Profile:**
    {query_text}

    **Relevant Products (from our knowledge base):**
    1. {retrieved_docs[0]}
    2. {retrieved_docs[1]}
    3. {retrieved_docs[2]}

    **Your Task:**
    Based on all of this information, what is the single best product to recommend to this customer? 
    Provide a concise, one-paragraph justification explaining *why* it's the best fit, referencing the customer's profile.
    
    **Recommendation:**
    """
    
    print("\nSending request to Groq LLM for final recommendation...")
    
    # CHANGE 3: Call the Groq API instead of the OpenAI API
    response = groq_client.chat.completions.create(
        # We use Llama 3, a powerful and popular open-source model
        model="llama3-8b-8192", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    
    return response.choices[0].message.content

def main():
    """
    Main function to run a demonstration of the recommendation engine.
    """
    index, model, product_texts = build_or_load_index()
    
    try:
        customer_df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{PROCESSED_DATA_PATH}'.")
        print("Please run 'process_data.py' first.")
        return

    # Let's test with a customer who might be a good candidate for a business loan.
    # You can change this ID to test other profiles.
    sample_customer_id = customer_df[customer_df['employment_status'] == 'Self-Employed'].iloc[0]['customer_id']
    
    recommendation = get_recommendation(sample_customer_id, customer_df, index, model, product_texts)
    
    print("\n--- FINAL RECOMMENDATION ---")
    print(recommendation)

if __name__ == "__main__":
    main()