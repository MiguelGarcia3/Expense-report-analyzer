# %%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from kneed import KneeLocator
from nltk.corpus import stopwords

# Initialize NLTK
nltk.download('stopwords', quiet=True)

st.title("🔧 Expense Analyzer Tool")
st.write("Upload your CSV file below and set parameters to analyze expenses.")

# File Upload
uploaded_file = st.file_uploader("Upload your file (.csv or .xlsx)", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Check file type and read accordingly
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    # Clean Text
    custom_stopwords = set(stopwords.words('english')).union({
        'pcs', 'piece', 'spare', 'component', 'material', 'set'
    })

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s\-\/]', '', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in custom_stopwords]
        return ' '.join(tokens)

    df['CleanText'] = df['Text'].apply(clean_text)
    df['CleanDescription'] = df['Material Description'].apply(clean_text)

     # Replace "multiple" in Material column with NaN or empty string
    df['Material'] = df['Material'].astype(str).str.strip()  # Normalize strings
    df['Material'] = df['Material'].replace(r'(?i)^multiple$', np.nan, regex=True)
    
    # Date and Amount Processing
    df['Document Date'] = pd.to_datetime(df['Document Date'])
    df['Amount in local currency'] = pd.to_numeric(df['Amount in local currency'], errors='coerce')
    df["YearMonth"] = df["Document Date"].dt.to_period("M")
    df = df[df['Amount in local currency'] > 0]

    # Keywords Input
    keywords_input = st.text_input("Enter keywords to filter out (comma-separated) add or remove words from the pre-set list:",
                                   "shipping, expedite, fee, service, freight, repair, overnight, next day, after hours, delivery")
    keywords = [kw.strip().lower() for kw in keywords_input.split(',')]
    pattern = '|'.join([r'\b' + re.escape(keyword) + r'\b' for keyword in keywords])
    df['CleanText'] = df['CleanText'].str.lower()
    initial_row_count = len(df)
    df = df[~df['CleanText'].str.contains(pattern, na=False, regex=True)]

    # Remove empty rows
    conditions_to_remove = (
        (df['Text'].isna() | (df['Text'].str.strip() == "")) &
        (df['Material Description'].isna() | (df['Material Description'].str.strip() == "")) &
        (df['Material'].isna())
    )
    df = df[~conditions_to_remove]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['CleanText'])

    # Clustering
    clustering = DBSCAN(eps=0.47, min_samples=2, metric='cosine')
    df['Cluster'] = clustering.fit_predict(X)

    # Generate new material numbers
    def generate_unique_id(row):
        if row['Cluster'] == -1:
            return f"NEW-M#-{str(row.name).zfill(5)}"
        else:
            return f"NEW-M{str(row['Cluster']).zfill(5)}"

    df['Material'] = df.apply(
        lambda row: row['Material'] if pd.notnull(row['Material']) and row['Material'] != ''
        else generate_unique_id(row), axis=1
    )

    # Inventory Parameters
    months_range = st.slider("Select number of months for inventory range:", 1, 12, 3)
    critical_price_threshold = st.number_input("Price threshold for critical parts ($):", 0.0, 100000.0, 100.0)
    critical_order_threshold = st.number_input("Enter order count threshold for critical parts:", min_value=1, value=5, step=1)

    # Flag single orders
    df['order_count'] = df.groupby('Material')['Material'].transform('count')
    df['Single_order_flag'] = df['order_count'] == 1

    single_order_df = df[df['Single_order_flag']].copy()
    df = df[~df['Single_order_flag']].copy()

    monthly_orders = df.groupby(["Material", "YearMonth"]).size().reset_index(name="Orders")
    monthly_orders["MonthStart"] = monthly_orders["YearMonth"].dt.to_timestamp()

    results = []
    for material, group in monthly_orders.groupby("Material"):
        group = group.sort_values("MonthStart")
        group.set_index("MonthStart", inplace=True)
        resampled = group["Orders"].resample("MS").sum()
        rolling_sum = resampled.rolling(window=months_range).sum().dropna()

        if rolling_sum.empty:
            max_inv = 1
            avg_inv = 1
        else:
            max_inv = int(rolling_sum.max())
            avg_inv = round(rolling_sum.mean())

        results.append({
            "Material": material,
            "Min Inventory": avg_inv,
            "Max Inventory": max_inv
        })

    inventory_stats = pd.DataFrame(results)

    # Merge back single order flags
    single_order_flag = single_order_df[['Material', 'Single_order_flag']]
    inventory_stats = inventory_stats.merge(single_order_flag, how='outer')
    inventory_stats['Single_order_flag'] = inventory_stats['Single_order_flag'].notna()
    inventory_stats.loc[inventory_stats['Single_order_flag'], 'Min Inventory'] = 1
    inventory_stats.loc[inventory_stats['Single_order_flag'], 'Max Inventory'] = 1

    df = pd.concat([df, single_order_df], ignore_index=True)
    df["Year"] = df["Document Date"].dt.year


    # Mark Critical Parts (if high cost OR high demand)
    high_cost = df[df["Amount in local currency"] >= critical_price_threshold]["Material"].unique()
    order_counts = df.groupby("Material").size().reset_index(name="Total Orders")
    high_demand = order_counts[order_counts["Total Orders"] >= critical_order_threshold]["Material"].unique()

    # Use set union for OR logic
    critical_parts = set(high_cost).union(set(high_demand))

    inventory_stats["Critical Part"] = inventory_stats["Material"].isin(critical_parts)
    inventory_stats["Critical Part"] = inventory_stats["Critical Part"].map({True: "Critical", False: "Noncritical"})
   
    # Add description, cost & cost center
    material_text_mapping = df.groupby('Material')['CleanText'].first()
    inventory_stats["Text"] = inventory_stats["Material"].map(material_text_mapping)
    inventory_stats.fillna({"Text": "N/A"}, inplace=True)

    inventory_stats["Amount in local currency"] = inventory_stats["Material"].map(
        df.groupby("Material")["Amount in local currency"].mean()
    )

    material_description_mapping = df.groupby('Material')['CleanDescription'].first()
    inventory_stats["Material Description"] = inventory_stats["Material"].map(material_description_mapping)
    inventory_stats.fillna({"Material Description": "N/A"}, inplace=True)

    inventory_stats["Cost Center Name"] = inventory_stats["Material"].map(
        df.groupby("Material")["Cost center name"].first()
    )
    
    # Output Final Excel
    inventory_stats.to_excel("expenses_stats.xlsx", index=False, engine='openpyxl')

    # Show success
    st.success("Analysis complete!")
    st.dataframe(inventory_stats.head())

    # Download button
    with open("expenses_stats.xlsx", "rb") as f:
        st.download_button("Download Results", data=f, file_name="expenses_stats.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
