import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # Remove extra spaces
    df.columns = df.columns.str.lower()  # Convert to lowercase
    return df

def calculate_score(row, age, location, job_title):
    score = 0
    
    if 'age' in row and str(row['age']) == age:
        score += 10
    
    if 'location' in row and row['location'].lower() == location.lower():
        score += 15
    elif 'location' in row and location.lower() in row['location'].lower():
        score += 5
    
    if 'job title' in row and row['job title'].lower() == job_title.lower():
        score += 25
    
    return score

def score_all_candidates(df, age, location, job_title):
    required_columns = ['skills set', 'age', 'location', 'job title']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Required columns missing: {', '.join(missing_columns)}. Please check the dataset.")
        return pd.DataFrame()
    
    df['score'] = df.apply(lambda row: calculate_score(row, age, location, job_title), axis=1)
    df = df.sort_values(by='score', ascending=False)
    
    return df

# Streamlit UI
st.title("📄 Resume Shortlisting System")

uploaded_file = st.file_uploader("Upload Resume Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Uploaded Dataset:")
    st.dataframe(df.head())
    st.write("Dataset Columns:", df.columns.tolist())  # Display column names
    
    age = st.text_input("Enter Age for Filtering:")
    location = st.text_input("Enter Location for Filtering:")
    job_title = st.text_input("Enter Required Job Title:")
    
    if st.button("Show All Candidates with Scores"):
        scored_df = score_all_candidates(df, age, location, job_title)
        
        if not scored_df.empty:
            st.write("### All Candidates with Scores:")
            st.dataframe(scored_df[['skills set', 'age', 'location', 'job title', 'score']])
            
            highest_score_candidate = scored_df.iloc[0]
            st.write("### Highest Scoring Candidate:")
            st.write(highest_score_candidate.to_frame().T)
            
            # Generate Bar Chart
            st.write("### Score Distribution of Candidates")
            plt.figure(figsize=(10, 5))
            plt.bar(scored_df.index, scored_df['score'], color='skyblue')
            plt.xlabel("Candidate Index")
            plt.ylabel("Score")
            plt.title("Scores of Candidates")
            st.pyplot(plt)
            
            # Generate Pie Chart for Location Distribution
            st.write("### Candidate Distribution by Location")
            location_counts = scored_df['location'].value_counts()
            plt.figure(figsize=(8, 8))
            plt.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
            plt.title("Distribution of Candidates by Location")
            st.pyplot(plt)
        else:
            st.warning("No candidates found.")
