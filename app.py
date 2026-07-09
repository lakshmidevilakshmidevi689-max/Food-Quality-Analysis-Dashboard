
# ==========================================
# FOOD QUALITY ANALYSIS DASHBOARD
# ==========================================

# INSTALL REQUIRED LIBRARIES:
# pip install streamlit pandas plotly scikit-learn

# RUN COMMAND:
# streamlit run app.py

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Food Quality Dashboard",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("🍎 Food Quality Analysis Dashboard")
st.write("Comprehensive Analysis of Food Quality, Safety & Nutrition")

st.markdown("---")

# ==========================================
# LOAD DATASET
# ==========================================

try:
    df = pd.read_csv("food_quality_dataset.csv")
except FileNotFoundError:
    st.error("Dataset file 'food_quality_dataset.csv' not found.")
    st.stop()

# ==========================================
# DATA CLEANING
# ==========================================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values safely
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].mean()
)

# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "Category",
    "Region",
    "Quality_Rating",
    "Bacterial_Count",
    "Price",
    "Food_Name",
    "Storage_Temperature",
    "Inspection_Result",
    "Moisture_Content",
    "pH_Level",
    "Fat_Content",
    "Protein_Content",
    "Sugar_Level",
    "Shelf_Life"
]

missing_cols = [
    col for col in required_columns
    if col not in df.columns
]

if missing_cols:
    st.error(f"Missing columns in dataset: {missing_cols}")
    st.stop()

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.title("Filters")

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + list(df["Category"].dropna().unique())
)

region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + list(df["Region"].dropna().unique())
)

# Apply Filters

filtered_df = df.copy()

if category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == category
    ]

if region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == region
    ]

# ==========================================
# KPI SECTION
# ==========================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Products", len(filtered_df))

col2.metric(
    "Average Quality",
    round(filtered_df["Quality_Rating"].mean(), 2)
)

col3.metric(
    "Average Bacteria",
    round(filtered_df["Bacterial_Count"].mean(), 2)
)

col4.metric(
    "Average Price",
    round(filtered_df["Price"].mean(), 2)
)

st.markdown("---")

# ==========================================
# CLEANED DATASET
# ==========================================

st.header("✅ Cleaned Dataset")

st.dataframe(filtered_df)

# ==========================================
# EDA REPORT
# ==========================================

st.header("📊 EDA Report")

# ------------------------------------------
# CATEGORY DISTRIBUTION
# ------------------------------------------

fig1 = px.histogram(
    filtered_df,
    x="Category",
    color="Category",
    title="Food Category Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------
# QUALITY RATING
# ------------------------------------------

fig2 = px.bar(
    filtered_df,
    x="Food_Name",
    y="Quality_Rating",
    color="Category",
    title="Food Quality Ratings"
)

st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------
# STORAGE TEMPERATURE VS BACTERIA
# ------------------------------------------

fig3 = px.scatter(
    filtered_df,
    x="Storage_Temperature",
    y="Bacterial_Count",
    color="Category",
    size="Quality_Rating",
    hover_name="Food_Name",
    title="Storage Temperature vs Bacterial Count"
)

st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------
# INSPECTION RESULT
# ------------------------------------------

fig4 = px.pie(
    filtered_df,
    names="Inspection_Result",
    title="Inspection Result Distribution"
)

st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# STATISTICAL SUMMARY
# ==========================================

st.header("📈 Statistical Summary")

st.dataframe(filtered_df.describe())

# ==========================================
# MACHINE LEARNING SECTION
# ==========================================

st.header("🤖 Predictive Model Results")

# Copy dataframe for ML
ml_df = filtered_df.copy()

# Convert target values safely
ml_df["Inspection_Result"] = ml_df["Inspection_Result"].astype(str)

ml_df["Inspection_Result"] = ml_df["Inspection_Result"].map({
    "Pass": 1,
    "Fail": 0
})

# Remove missing mapped values
ml_df = ml_df.dropna(subset=["Inspection_Result"])

# Features
X = ml_df[[
    "Storage_Temperature",
    "Moisture_Content",
    "pH_Level",
    "Fat_Content",
    "Protein_Content",
    "Sugar_Level",
    "Bacterial_Count",
    "Shelf_Life",
    "Quality_Rating"
]]

# Target
y = ml_df["Inspection_Result"]

# Check minimum rows
if len(ml_df) > 5:

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    # Prediction
    prediction = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, prediction)

    st.success(f"Model Accuracy: {accuracy*100:.2f}%")

else:
    st.warning("Not enough data for machine learning.")

# ==========================================
# BUSINESS INSIGHTS
# ==========================================

st.header("💡 Business Insights")

st.write("✔ Dairy products have better quality ratings.")
st.write("✔ High bacterial count reduces food quality.")
st.write("✔ Proper storage temperature improves shelf life.")
st.write("✔ Products with high contamination fail inspection.")
st.write("✔ Vegetables and Meat categories need strict monitoring.")

# ==========================================
# FOOD SAFETY RECOMMENDATIONS
# ==========================================

st.header("🛡 Food Safety Recommendations")

st.write("✔ Maintain proper cold storage systems.")
st.write("✔ Monitor bacterial contamination regularly.")
st.write("✔ Improve food packaging quality.")
st.write("✔ Conduct frequent food inspections.")
st.write("✔ Use smart sensors for temperature monitoring.")

# ==========================================
# FINAL CONCLUSION REPORT
# ==========================================

st.header("📄 Final Conclusion Report")

st.write("""
This Food Quality Analysis Dashboard analyzes
food quality, contamination, safety, and nutritional trends
using Data Analysis, Visualization, and Machine Learning.

The dashboard helps identify:
- Unsafe food products
- High contamination risks
- Quality trends
- Storage condition impacts

This project can be used for:
- Academic Projects
- Mini Projects
- Major Projects
- Research Work
- Industry-Level Food Monitoring Systems
""")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Developed using Python, Streamlit, Plotly & Machine Learning"
)