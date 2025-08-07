import streamlit as st
import pandas as pd
from recommendation_engine import build_or_load_index, get_recommendation

# --- Page Configuration ---
st.set_page_config(
    page_title="COMPASS Recommendation Engine",
    page_icon="🧭",
    layout="wide"
)

# --- NEW: CSS for Word Wrapping ---
# This injects a custom CSS style into the Streamlit app.
# The .word-wrap-text class will force long, unbroken strings to wrap correctly.
st.markdown("""
    <style>
    .word-wrap-text {
        overflow-wrap: break-word;
        word-wrap: break-word; /* For older browser compatibility */
        word-break: break-all;  /* Force breaks even in the middle of a word */
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    """
    Load all the necessary models and data.
    """
    print("Loading resources for the first time...")
    customer_df = pd.read_csv('processed_customer_data.csv')
    index, model, product_texts = build_or_load_index()
    print("Resources loaded successfully.")
    return customer_df, index, model, product_texts

# --- Main Application ---
customer_df, index, model, product_texts = load_resources()

st.title("🧭 COMPASS: Customer Opportunity Mapping")
st.write(
    "This tool uses an AI-powered recommendation engine to identify the best cross-sell opportunities for banking customers. "
    "Enter a Customer ID below to get a personalized product recommendation."
)

st.divider()

st.subheader("Get a Customer Recommendation")
col1, col2 = st.columns([1, 3])

with col1:
    customer_id_input = st.number_input(
        "Enter Customer ID:", 
        min_value=1, 
        max_value=len(customer_df), 
        value=125,
        step=1
    )

with col2:
    st.write("") 
    st.write("")
    if st.button("Generate Recommendation", type="primary", use_container_width=True):
        
        if customer_id_input:
            customer_id = int(customer_id_input)
            
            with st.spinner(f"Analyzing Customer {customer_id} and generating recommendation..."):
                customer_data = customer_df[customer_df['customer_id'] == customer_id]
                recommendation_text = get_recommendation(customer_id, customer_df, index, model, product_texts)

                st.divider()
                st.subheader("Recommendation Report")
                
                st.write("**Customer Snapshot:**")
                st.dataframe(customer_data)
                
                st.success("**AI-Powered Recommendation:**")
                
                # --- CHANGED: Apply the CSS style to the output text ---
                # We use st.markdown to create a div with our custom class, ensuring the text wraps correctly.
                # The 'unsafe_allow_html=True' is necessary to render the HTML div.
                st.markdown(f'<div class="word-wrap-text">{recommendation_text}</div>', unsafe_allow_html=True)
        else:
            st.error("Please enter a valid Customer ID.")