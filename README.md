# 🧭 COMPASS: Customer Opportunity Mapping (Methodology Replica)

This repository contains a functional replica of the COMPASS system, an AI-powered recommendation engine designed for the banking industry. The original project was developed using proprietary customer data to drive cross-sell conversions. This version faithfully replicates the project's architecture and logic using synthetically generated data.

The core of this project is a sophisticated **Retrieval-Augmented Generation (RAG)** pipeline that analyzes customer profiles to identify and justify the next best product recommendation, demonstrating a powerful approach to data-driven customer relationship management.

![Streamlit App Screenshot](https://github.com/code2ashish/-COMPASS-Customer-Opportunity-Mapping/blob/main/Screenshot.png) 

---

## ✨ Project Philosophy

The primary goal of this repository is to showcase the *methodology* behind building a production-grade recommendation system. While the original project utilized a fine-tuned, proprietary language model and was deployed on a larger cloud infrastructure, this replica uses accessible, state-of-the-art open-source components to achieve a similar outcome, proving the viability of the architecture.

## 🛠️ Tech Stack & Architecture

This replica demonstrates a robust technical architecture. The production system it mirrors would leverage a similar, but more scaled, stack.

| Component | This Replica's Implementation | Production-Grade Target |
| :--- | :--- | :--- |
| **Data Source** | Synthetically Generated CSVs | Real-time Customer Data Lake (e.g., on AWS S3) |
| **Feature Engineering** | In-Memory SQLite via SQLAlchemy | Scheduled PySpark Jobs on Databricks/AWS EMR |
| **Embedding Model** | `all-MiniLM-L6-v2` (for speed) | **`bge-large-en-v1.5`** (for state-of-the-art semantic understanding) |
| **Vector Database** | **FAISS** (Local File Index) | **Pinecone / Weaviate** (Managed Cloud Vector DB) |
| **Generative Model** | **Llama 3** (via Groq API) | **A fine-tuned Mistral-7B model** (for domain-specific accuracy) |
| **Frontend** | Streamlit | A custom React frontend integrated into a CRM |

---

## 🚀 Local Setup and Execution

**Important:** This repository does not contain the large data files. You must generate them locally by running the provided scripts.

**1. Clone the repository:**
```bash
git clone https://github.com/code2ashish/-COMPASS-Customer-Opportunity-Mapping.git
cd -COMPASS-Customer-Opportunity-Mapping
```

**2. Install the required dependencies:**
```bash
pip install pandas numpy faker sqlalchemy sentence-transformers faiss-cpu groq streamlit
```

**3. Set Up Your Groq API Key:**
This project uses the free and fast Groq API to power the LLM.
*   Get your free key from [GroqCloud](https://console.groq.com/).
*   Set it as an environment variable named `GROQ_API_KEY`.

On Windows (Command Prompt):
```cmd
setx GROQ_API_KEY "your_groq_key_goes_here"
```
*(Remember to close and reopen your terminal after setting the key.)*

---

## ▶️ Running the Application

The scripts must be executed in the following order:

**1. Step 1: Generate Raw Customer Data**
This script creates `customer_data.csv` (approx. 55 MB).
```bash
python generate_data.py
```

**2. Step 2: Process Data and Engineer Features**
This script reads the raw data and creates the feature-rich `processed_customer_data.csv` (approx. 65 MB).
```bash
python process_data.py```

**3. Step 3: Launch the COMPASS Dashboard**
This will start the Streamlit web server and open the application in your browser. The first time you run it, it will also build the FAISS vector index (`faiss_index.bin`).
```bash
streamlit run app.py
```

## 📂 Project Structure

```
COMPASS_Project/
├── generate_data.py          # Script to create synthetic customer data.
├── process_data.py           # Script to clean data and engineer features.
├── recommendation_engine.py  # Core RAG logic for generating recommendations.
├── app.py                    # The Streamlit frontend application.
├── products.txt              # The knowledge base for the RAG system.
├── .gitignore                # Specifies files for Git to ignore (like *.csv).
└── README.md                 # This project description file.
```
