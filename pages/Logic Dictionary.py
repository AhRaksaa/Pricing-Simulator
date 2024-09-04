import streamlit as st
import pandas as pd


with open('style/logo.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown('<h1 style="color: #205527;">Logic Words\' Search</h1>', unsafe_allow_html=True)

# Load your dataset (replace 'data.csv' with your actual file)
df = pd.read_csv('file/logic_dic.csv')

st.write("")
st.write("")
st.dataframe(df)
st.write("")
st.write("")


# Input field for the keyword
keyword = st.text_input("Enter a keyword")

# Search button
if st.button("Search"):
    # Filter the DataFrame based on the keyword (adjust filtering logic as needed)
    results = df[df['key_word'].str.contains(keyword, case=False)]

    # Display the results
    if results.empty:
        st.warning("No results found.")
    else:
        st.subheader("Search Results")
        st.dataframe(results)

