import pandas as pd
import streamlit as st
import numpy as np
import openai
import seaborn as sns
import matplotlib.pyplot as plt

with open('style/logo.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown('<h1 style="color: #205527;">Upload Your Data</h1>', unsafe_allow_html=True)

# Upload CSV data
with st.header('1. Upload your CSV data'):
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv

    df = load_csv()
    st.header('**Input DataFrame**')
    st.write(df.head(10))
    st.write('---')
    st.header('**Data Exploration**')

    # Display DataFrame information in tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Columns", "Null Values", "Duplicates", "Describe", "Shape", "Data Types", "Correlation", "Outliers"])

    tab1.text("Your Dataset's Columns")
    tab1.write(df.columns)

    tab2.text("How many missing values does this dataframe have?")
    tab2.write(df.isnull().sum())

    tab3.text("Discover Duplicates ")
    tab3.write(df.duplicated().sum())

    tab4.text("Descriptive Statistics")
    tab4.write(df.describe())

    tab5.text("Your Data Shape")
    tab5.write(df.shape)

    tab6.text("Identify Data Types, Formats, and Structures ")
    tab6.write(df.dtypes)

    tab7.text("Calculate correlations between numerical variables to identify potential relationships")
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype('category')

    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    correlation = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="viridis", ax=ax)
    plt.title("Correlation Heatmap")
    tab7.pyplot(fig)

    tab8.text("Identify Outliers")
    column = tab8.selectbox("Select a column:", df.columns)

    def identify_outliers_iqr(df, column):
        if pd.api.types.is_numeric_dtype(df[column]):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            return outliers
        else:
            return pd.DataFrame()

    outliers = identify_outliers_iqr(df, column)

    if outliers.empty:
        tab8.write("No outliers found.")
    else:
        tab8.write("Outliers:")
        tab8.dataframe(outliers)

    if pd.api.types.is_numeric_dtype(df[column]):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(df[column])
        plt.title(f"Box Plot for {column}")
        plt.xlabel(column)
        plt.ylabel("Values")
        tab8.pyplot(fig)
    else:
        tab8.write("Box plot can only be created for numeric columns.")

    # User question input and submit button
    user_question = st.chat_input("Ask a question about the data:")
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        openai.api_key = api_key

        dataset_summary = df.to_dict(orient='records')

        prompt = f"""Here's some data: {dataset_summary}

        Please answer the following question based on the provided data: 
        {user_question}
        """

        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        ai_response = completion.choices[0].message['content']
        st.write("AI Answer:")
        st.write(ai_response)
    except KeyError:
        st.error("Please set the environment variable 'OPENAI_API_KEY' with your API key")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    if not user_question:
        st.info("Upload a CSV file to preview its data.")