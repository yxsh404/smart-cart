import sqlite3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load Datasets
customer_df = pd.read_csv("customer_data_collection.csv")
product_df = pd.read_csv("product_recommendation_data.csv")

# Remove unnamed columns
customer_df = customer_df.loc[:, ~customer_df.columns.str.contains('^Unnamed')]
product_df = product_df.loc[:, ~product_df.columns.str.contains('^Unnamed')]

# Print available columns to debug
print("Customer Columns:", customer_df.columns)
print("Product Columns:", product_df.columns)

# Rename column for consistency
if "Similar_Product_List" in product_df.columns:
    product_df.rename(columns={"Similar_Product_List": "Similar_Products"}, inplace=True)

# Identify categorical columns, ensuring they exist
customer_categorical_cols = [col for col in ['Gender', 'Location', 'Customer_Segment', 'Holiday', 'Season'] if col in customer_df.columns]
product_categorical_cols = [col for col in ['Category', 'Subcategory', 'Brand', 'Holiday', 'Season', 'Geographical_Location'] if col in product_df.columns]

# Apply one-hot encoding to categorical columns
customer_df = pd.get_dummies(customer_df, columns=customer_categorical_cols, drop_first=True)
product_df = pd.get_dummies(product_df, columns=product_categorical_cols, drop_first=True)

# Apply frequency encoding to high-cardinality categorical columns if they exist
high_cardinality_cols = ['Browsing_History', 'Purchase_History']
for col in high_cardinality_cols:
    if col in customer_df.columns:
        freq_map = customer_df[col].value_counts().to_dict()
        customer_df[col] = customer_df[col].map(freq_map)

# Apply MinMaxScaler only to numerical columns
scaler = MinMaxScaler()
customer_numerical_cols = customer_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
product_numerical_cols = product_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if customer_numerical_cols:
    customer_df[customer_numerical_cols] = scaler.fit_transform(customer_df[customer_numerical_cols])
if product_numerical_cols:
    product_df[product_numerical_cols] = scaler.fit_transform(product_df[product_numerical_cols])

# Process Similar_Products column safely
if 'Similar_Products' in customer_df.columns:
    customer_df['Similar_Products'] = customer_df['Similar_Products'].apply(
        lambda x: x.strip("[]").replace("'", "").split(', ') if isinstance(x, str) else []
    )
    customer_df = customer_df.explode('Similar_Products')

if 'Similar_Products' in product_df.columns:
    product_df['Similar_Products'] = product_df['Similar_Products'].apply(
        lambda x: x.strip("[]").replace("'", "").split(', ') if isinstance(x, str) else []
    )
    product_df = product_df.explode('Similar_Products')

# Save cleaned datasets
customer_df.to_csv('customer_data_cleaned.csv', index=False)
product_df.to_csv('product_data_cleaned.csv', index=False)

# Store in SQLite database
conn = sqlite3.connect('ecommerce_data.db')
customer_df.to_sql("customers", conn, if_exists="replace", index=False)
product_df.to_sql("products", conn, if_exists="replace", index=False)
conn.close()

print("Data preprocessing complete. Cleaned files saved as CSV and SQLite database updated.")
