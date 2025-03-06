import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="Resume Shortlisting & Scoring Dashboard", layout="wide")

# Load dataset
def load_data(file):
    # Read the uploaded file into a dataframe
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

# Preprocess data
def preprocess_data(df):
    # Drop unnecessary columns
    df.drop(columns=["full_name", "phone_number", "email"], errors="ignore", inplace=True)
    
    # Ensure salary is in numeric format (no scaling applied)
    df["salary"] = pd.to_numeric(df["current_ctc"], errors="coerce")
    
    # Fill missing salary values with median
    df["salary"].fillna(df["salary"].median(), inplace=True)
    
    # Optionally, round salary values to nearest integer to avoid decimals
    df["salary"] = df["salary"].round().astype(int)
    
    # Drop the 'current_ctc' column as it's no longer needed
    df.drop(columns=["current_ctc"], inplace=True)
    
    return df

# Upload file using streamlit's file uploader
st.title("📄 Resume Shortlisting & Scoring Dashboard")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        df = preprocess_data(df)

        # Sidebar Filters - These will only be available after uploading a file
        with st.sidebar:
            st.header("🔍 Filter Candidates")
            
            # Select Job Titles dynamically
            if "Job Title" in df.columns:
                job_titles = df["Job Title"].dropna().unique()
                selected_jobs = st.multiselect("Select Job Titles", job_titles)

            # Select Locations dynamically
            if "Location" in df.columns:
                locations = df["Location"].dropna().unique()
                selected_locations = st.multiselect("Select Locations", locations)

            # Select Skills Set dynamically
            if "Skills Set" in df.columns:
                skills_set = df["Skills Set"].dropna().unique()
                selected_skills = st.multiselect("Select Skills Set", skills_set)

            # Ensure valid experience range dynamically
            if "experience" in df.columns and not df["experience"].dropna().empty:
                experience_min = int(df["experience"].min())
                experience_max = int(df["experience"].max())
            else:
                experience_min, experience_max = 0, 10  # Default safe range

            experience_range = st.slider("Select Experience Range", experience_min, experience_max, (experience_min, experience_max))

            # Ensure valid salary range dynamically
            if "salary" in df.columns and not df["salary"].dropna().empty:
                salary_min = int(df["salary"].min()) if df["salary"].min() < df["salary"].max() else 0
                salary_max = int(df["salary"].max()) if df["salary"].min() < df["salary"].max() else 10000000
            else:
                salary_min, salary_max = 0.0, 10.0  # Default safe range

            salary_range = st.slider("Select Salary Range", min_value=salary_min, max_value=salary_max, value=(salary_min, salary_max), step=1000)
        
        # Filtering Candidates
        filtered_df = df.copy()
        
        if selected_jobs:
            filtered_df = filtered_df[filtered_df["Job Title"].isin(selected_jobs)]
        
        if selected_locations:
            filtered_df = filtered_df[filtered_df["Location"].isin(selected_locations)]
        
        if selected_skills:
            filtered_df = filtered_df[filtered_df["Skills Set"].apply(lambda x: any(skill in x for skill in selected_skills) if isinstance(x, str) else False)]
        
        filtered_df = filtered_df[filtered_df["experience"].between(experience_range[0], experience_range[1])]
        filtered_df = filtered_df[filtered_df["salary"].between(salary_range[0], salary_range[1])]
        
        # Apply RandomForest Model
        if not filtered_df.empty:
            feature_columns = ["experience", "salary"]
            X = filtered_df[feature_columns]
            y = np.random.randint(0, 2, size=len(filtered_df))  # Placeholder target variable

            # Check if there are enough samples to split
            if len(filtered_df) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                predictions = rf_model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                st.write(f"Random Forest Model Accuracy: {accuracy:.2f}")
            else:
                st.warning("⚠ Not enough data to train the model. Skipping model training.")
        
        # Display filtered results
        if filtered_df.empty:
            st.warning("⚠ No candidates match the selected filters. Try adjusting the filters.")
        else:
            st.dataframe(filtered_df, height=400)
            
            # Visualizations
            st.subheader("📊 Candidate Analysis")
            
            # Bar Chart for Top Candidates
            fig = px.bar(filtered_df.head(10), x="Job Title", y="salary", color="salary", title="Top 10 Candidates by Salary")
            st.plotly_chart(fig, use_container_width=True)
            
            # Pie Chart for Location Distribution
            st.subheader("🌍 Candidate Distribution by Location")
            location_counts = filtered_df["Location"].value_counts()
            if not location_counts.empty:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
                ax.set_title("Distribution of Candidates by Location")
                st.pyplot(fig)
            else:
                st.warning("⚠ No location data available for pie chart.")
        
        st.success("✅ Dashboard Ready!")
else:
    st.warning("⚠ Please upload a CSV file to get started.")
