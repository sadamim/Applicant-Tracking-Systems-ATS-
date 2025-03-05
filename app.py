import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="Resume Shortlisting & Scoring Dashboard", layout="wide")

# Load dataset
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    df = pd.read_csv(file_path)
    return df

# Preprocess data
def preprocess_data(df):
    df.drop(columns=["full_name", "phone_number", "email"], errors="ignore", inplace=True)
    df.fillna({"current_ctc?": "Unknown"}, inplace=True)

    # Convert salary to numerical format
    def parse_salary(salary):
        if isinstance(salary, str):
            salary = salary.lower().replace("lakhs", "00000").replace("k", "000").replace("lac", "00000")
            salary = ''.join(filter(str.isdigit, salary))
            return float(salary) if salary else np.nan
        return np.nan

    df["salary"] = df["current_ctc?"].apply(parse_salary)
    df.drop(columns=["current_ctc?"], inplace=True)
    df["salary"].fillna(df["salary"].median(), inplace=True)
    scaler = StandardScaler()
    df["salary"] = scaler.fit_transform(df[["salary"]])
    
    return df

# Load data
file_path = "resume_data.csv"
df = load_data(file_path)
if df is not None:
    df = preprocess_data(df)
    
    st.title("📄 Resume Shortlisting & Scoring Dashboard")
    
    # Sidebar Filters
    with st.sidebar:
        st.header("🔍 Filter Candidates")
        job_titles = df["Job Title"].unique()
        selected_jobs = st.multiselect("Select Job Titles", job_titles, job_titles[:1])
        locations = df["Location"].unique()
        selected_locations = st.multiselect("Select Locations", locations, locations[:1])
        skills_set = df["Skills Set"].unique()
        selected_skills = st.multiselect("Select Skills Set", skills_set, skills_set[:1])
        salary_range = st.slider("Select Salary Range", float(df["salary"].min()), float(df["salary"].max()), (float(df["salary"].min()), float(df["salary"].max())))
    
    # Filtering Candidates
    filtered_df = df[(df["Job Title"].isin(selected_jobs)) &
                     (df["Location"].isin(selected_locations)) &
                     (df["Skills Set"].isin(selected_skills)) &
                     (df["salary"].between(salary_range[0], salary_range[1]))]
    
    # Scoring System
    def calculate_score(row):
        return np.random.randint(10, 100)  # Introduce variability
    
    if not filtered_df.empty:
        filtered_df["Score"] = filtered_df.apply(calculate_score, axis=1)
        filtered_df = filtered_df.sort_values(by="Score", ascending=False)
    
    # Display Filtered Resumes
    st.subheader("🎯 Filtered & Scored Resumes")
    if not filtered_df.empty:
        st.dataframe(filtered_df, height=400)
        
        # Bar Chart for Top Candidates by Score using Plotly
        st.subheader("📊 Top Candidates by Score")
        fig = px.bar(filtered_df.head(10), x="Job Title", y="Score", color="Score", title="Top 10 Candidates by Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar Chart for Top Candidates using Matplotlib
        st.subheader("📊 Top Candidates by Score (Matplotlib)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(filtered_df["Job Title"].head(10), filtered_df["Score"].head(10), color="skyblue")
        ax.set_xlabel("Score")
        ax.set_ylabel("Job Title")
        ax.set_title("Top 10 Candidates by Score")
        ax.invert_yaxis()  # Invert y-axis for ranking order
        st.pyplot(fig)
        
        # Pie Chart for Location Distribution using Matplotlib
        st.subheader("🌍 Candidate Distribution by Location")
        location_counts = filtered_df["Location"].value_counts()
        if not location_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
            ax.set_title("Distribution of Candidates by Location")
            st.pyplot(fig)
        
        
        # Scatter Plot for Salary vs Score
        st.subheader("💰 Salary vs Score Distribution")
        fig = px.scatter(filtered_df, x="salary", y="Score", color="Score", size="Score", title="Salary vs Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠ No candidates match the selected filters.")
    
    st.success("✅ Dashboard Ready!")