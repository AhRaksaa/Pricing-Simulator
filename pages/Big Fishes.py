import streamlit as st
import pandas as pd
import openai
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards

with open('style/logo.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Path to your Excel file
file_path = 'file/Update_Big_Fish_Logic.xlsx'

# Reading all sheets from the Excel file into a dictionary
sheets_dict = pd.read_excel(file_path, engine='openpyxl', sheet_name=None)

# Load HTML content from an external file
with open('style/template.html', 'r') as f:
    html_content = f.read()

# Display HTML content in Streamlit
st.markdown(html_content, unsafe_allow_html=True)

def store_question_history(question, answer):
    if 'question_history' not in st.session_state:
        st.session_state.question_history = {}
    st.session_state.question_history[question] = answer

def delete_question_history():
    if 'question_history' in st.session_state:
        del st.session_state.question_history

# Display EDA 
col1, col2, col3, col4 = st.columns(4)

col1.text("WHS-ABC")
col1.image("https://images.ctfassets.net/w6n4s696snx1/6oWXFRgIsRFhtUBFZ9Wkf0/bad2746aa7452fb4543e64f2f3251288/cropped-ABC-Logo-PNG-150x150.webp?w=480", width=80)
col1.metric("Total Outlests","135","")
col1.metric("Volume Drop(KHL)","-9.76","-123K ctn")

col2.text("WHS-TIGER")
col2.image("https://pbs.twimg.com/profile_images/1220812008712540163/sXtES8E1_400x400.jpg", width=80)
col2.metric("Total Outlests","83","")
col2.metric("Volume Drop(KHL)","-7.2","-91K ctn")

col3.text("DS-ABC")
col3.image("https://images.ctfassets.net/w6n4s696snx1/6oWXFRgIsRFhtUBFZ9Wkf0/bad2746aa7452fb4543e64f2f3251288/cropped-ABC-Logo-PNG-150x150.webp?w=480", width=80)
col3.metric("Total Outlests","773","")
col3.metric("Volume Drop(KHL)","-7.2","-90K ctn")

col4.text("DS-TIGER")
col4.image("https://pbs.twimg.com/profile_images/1220812008712540163/sXtES8E1_400x400.jpg", width=80)
col4.metric("Total Outlests","829","")
col4.metric("Volume Drop(KHL)","-13.9","-175K ctn")
style_metric_cards()

st.write("")
st.write("")
#st.markdown('<h3 style="color: #205527;">Trend Line</h3>', unsafe_allow_html=True)
with st.expander("Trend Line"):
    for sheet_name, df in sheets_dict.items():
        # Count the total number of outlets (rows) in the sheet
        total_outlets = df.shape[0]
        st.write(f"Total Outlets for {sheet_name}")
        
        # Sum the values in the columns 'Vol_2022 (Hl)', 'Vol_2023 (Hl)', 'Vol_2024 (Hl)'
        total_volume = df[['Vol_2022 (Hl)', 'Vol_2023 (Hl)', 'Vol_2024 (Hl)']].sum()
        
        # Create a line chart for total volume by year using Plotly Express
        #fig = px.line(x=total_volume.index, y=total_volume.values, markers=True, title=f'Total Volume by Year for {sheet_name}')
        fig = px.line(x=total_volume.index, y=total_volume.values, markers=True)
        fig.update_layout(xaxis_title='Year', yaxis_title='Total Volume')
        
        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig)

st.write("")
st.markdown('<h3 style="color: #205527;">Ask AI About The Insight:</h3>', unsafe_allow_html=True)
user_question = st.text_input("Enter your question:")

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

        store_question_history(user_question, ai_response)  # Store the user question and AI answer

    except KeyError:
        st.error("Please set the environment variable 'OPENAI_API_KEY' with your API key")
    except Exception as e:
        st.error(f"An error occurred: {e}")



if st.checkbox("Show Question History"):
    if 'question_history' in st.session_state:
        for idx, (question, answer) in enumerate(st.session_state.question_history.items(), 1):
            st.write(f"{idx}. Question: {question}")
            st.write(f"Answer: {answer}")
            
        if st.button("Delete Question History"):
            delete_question_history()
            st.write("Question history deleted.")
    else:
        st.write("No questions asked yet.")\

st.markdown('<h1 style="color: #205527;">Recommendation</h1>', unsafe_allow_html=True)
st.write("")
st.write("")
with open('style/rfm.html', 'r') as f:
    html_content_1 = f.read()


# Display HTML content in Streamlit
st.markdown(html_content_1, unsafe_allow_html=True)