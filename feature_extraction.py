import pandas as pd
from sklearn.preprocessing import StandardScaler
import ollama
from transformers import AutoTokenizer, AutoModel
import torch
import json

# Load preprocessed datasets
customer_df = pd.read_csv("customer_data_cleaned.csv")
product_df = pd.read_csv("product_data_cleaned.csv")

# Drop identifier columns not needed for medelling but keep for linking later
customer_features =  customer_df.copy()
customer_features.drop(columns=["Customer_ID"], inplace=True)

# Standardixe numerical features 
numerical_cols = customer_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
customer_features[numerical_cols] = scaler.fit_transform(customer_features[numerical_cols])

# Save custoemr featuers
customer_features.to_csv("customer_features.csv", index=False)
print("Done")

# For features we want both numerical features and semantic embedding of text attributes
product_features = product_df.copy()

# Combine key textual attributes to form descriptionn for embedding and adjust field names as needed
def combine_text_fields(row):
    fields = []
    for col in ['Category', 'Subcategory', 'Brand']:
        if col in row and pd.notnull(row[col]):
            fields.append(str(row[col])) 
    return " ".join(fields)    

# create a new column combining text attributes
product_features["Combined_Text"] = product_features.apply(combine_text_fields, axis=1)

# Initixalize an embedding model 
def get_gemma_embedding(text):
    response = ollama.embeddings(model="gemma:2b", prompt=text)
    return response["embedding"]

# Generate embeddings using Mistral
product_features["Embeddings"] = product_features["Combined_Text"].apply(get_gemma_embedding)

# Standardize numerical features
prod_num_cols = product_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
product_features[prod_num_cols] = scaler.fit_transform(product_features[prod_num_cols])

# Save product features
product_features.to_csv("product_features.csv", index=False)
print("Done")


