import pandas as pd
import streamlit as st
import openai

with open('style/logo.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def main():
    """Displays a file uploader and previews the uploaded CSV data."""
    
    st.markdown('<h1 style="color: #205527;">Data ChatBot</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV data
            df = pd.read_csv(uploaded_file, encoding='utf-8')

            # Display the first 5 rows of the DataFrame
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Ask a question or request insights:")
            user_question = st.text_input("Type your question or request here:")

            if st.button("Submit"):
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

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Upload a CSV file to preview its data.")

if __name__ == "__main__":
    main()