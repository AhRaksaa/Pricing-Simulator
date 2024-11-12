import streamlit as st
import pandas as pd
import numpy as np
import openai
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
import joblib  
import datetime
from sklearn.preprocessing import StandardScaler

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

col1.text("WHS-B")
#col1.image("https://images.ctfassets.net/w6n4s696snx1/6oWXFRgIsRFhtUBFZ9Wkf0/bad2746aa7452fb4543e64f2f3251288/cropped-ABC-Logo-PNG-150x150.webp?w=480", width=80)
col1.metric("Total Outlests","00","")
col1.metric("Volume Drop(KHL)","00","00K ctn")

col2.text("WHS-A")
#col2.image("https://pbs.twimg.com/profile_images/1220812008712540163/sXtES8E1_400x400.jpg", width=80)
col2.metric("Total Outlests","00","")
col2.metric("Volume Drop(KHL)","00","00K ctn")

col3.text("DS-B")
#col3.image("https://images.ctfassets.net/w6n4s696snx1/6oWXFRgIsRFhtUBFZ9Wkf0/bad2746aa7452fb4543e64f2f3251288/cropped-ABC-Logo-PNG-150x150.webp?w=480", width=80)
col3.metric("Total Outlests","00","")
col3.metric("Volume Drop(KHL)","00","00K ctn")

col4.text("DS-B")
#col4.image("https://pbs.twimg.com/profile_images/1220812008712540163/sXtES8E1_400x400.jpg", width=80)
col4.metric("Total Outlests","00","")
col4.metric("Volume Drop(KHL)","00","00K ctn")
style_metric_cards()

st.write("")
st.write("")
#st.markdown('<h3 style="color: #205527;">Trend Line</h3>', unsafe_allow_html=True)

# Line Graph
tab_labels = list(sheets_dict.keys())
tabs = st.tabs(tab_labels)

# style of streamlit tab 
st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 20px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 40px;
        color: #205527;
        width: 150px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #205527;
        color: #FFFFFF;
        
	}
</style>""", unsafe_allow_html=True)

for idx, (sheet_name, df) in enumerate(sheets_dict.items()):
    tab = tabs[idx]

    # Count the total number of outlets (rows) in the sheet
    total_outlets = df.shape[0]
    #st.write(f"Total Outlets for {sheet_name}: {total_outlets}")

    # Sum the values in the columns 'Vol_2022 (Hl)', 'Vol_2023 (Hl)', 'Vol_2024 (Hl)'
    total_volume = df[['Vol_2022 (Hl)', 'Vol_2023 (Hl)', 'Vol_2024 (Hl)']].sum()

    # Create a line chart for total volume by year using Plotly Express
    fig = px.line(x=total_volume.index, y=total_volume.values, markers=True, title=f'Total Volume by Year for {sheet_name}')
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Volume')

    # Display the Plotly figure in Streamlit
    tab.plotly_chart(fig)
        
# Recommendation Part
st.markdown('<h1 style="color: #205527;">Prediction</h1>', unsafe_allow_html=True)
st.write("")

# Load the Random Forest model using joblib
model = joblib.load('best_model/best_rf_model.pkl')  # Use joblib to load the Random Forest model

# Function to calculate if a date is a weekend
def weekend_or_weekday(date):
    day_number = date.weekday()
    return 1 if day_number >= 5 else 0

# Function to calculate the week of the month
def week_of_month(date):
    year, month, day = date.isocalendar()
    first_week_day = (day - date.day % 7 + 1) if date.day else 1
    return (date.day + first_week_day - 2) // 7 + 1

# Function to check if the month is in the specified seasons
def is_season(month):
    return month in [3, 4, 9, 11, 12]

# Function to check if the selected date is a holiday
def is_holiday(date):
    holidays = [
        datetime.date(date.year, 1, 1),    # New Year's Day
        datetime.date(date.year, 12, 25),  # Christmas Day
        datetime.date(date.year, 7, 4),    # Independence Day (USA)
        # Add other holidays as needed
    ]
    return 1 if date in holidays else 0

item_brand = st.selectbox("Select Item Brand", ["a", "b"])
item_sub_brand = st.selectbox("Select Item Sub-Brand", ["b_strong", "a_first", "a_diamond", "a_korea", "b_withhold"])
segment = st.selectbox("Select Segment", ["drink shop", "wholesaler"])
date_input = st.date_input("Select Date", datetime.date.today())

# Convert date input to datetime object
date = pd.to_datetime(date_input)

# Calculate additional features
year = date.year
month = date.month
day = date.day
is_weekend = weekend_or_weekday(date)
week_of_month_val = week_of_month(date)
is_season_val = is_season(month)
is_holiday_val = is_holiday(date)  # Calculate if it's a holiday

# Prepare the input data for prediction
data = pd.DataFrame({
    'year': [year],
    'month': [month],
    'day': [day],
    'is_weekend': [is_weekend],
    'week_of_month': [week_of_month_val],
    'is_season': [is_season_val],
    'is_holiday': [is_holiday_val],  # Added the is_holiday feature
    'item_brand_b': [1 if item_brand == 'b' else 0],
    'item_sub_brand_b_strong': [1 if item_sub_brand == 'b_strong' else 0],
    'item_sub_brand_a_diamond': [1 if item_sub_brand == 'a_diamond' else 0],
    'item_sub_brand_a_korea': [1 if item_sub_brand == 'a_korea' else 0],
    'item_sub_brand_b_withhold': [1 if item_sub_brand == 'b_withhold' else 0],
    'segment_wholesaler': [1 if segment == 'wholesaler' else 0]
})

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale the features using the StandardScaler
scaled_data = scaler.fit_transform(data)

# Button to trigger prediction
if st.button("Make Prediction"):
    # Predict using the trained Random Forest model
    prediction = model.predict(scaled_data)
    
    # Display the prediction as an integer
    st.write(f"Predicted Quantity: {int(prediction[0])}")

# OpenAI API 
st.write("")
st.write("")
st.write("")
#st.markdown('<h3 style="color: #205527;">Data ChatBot</h3>', unsafe_allow_html=True)

user_question = st.chat_input("Enter your question and press Enter to submit:")
if user_question:
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
        st.write("No questions asked yet.")

#with open('style/rfm.html', 'r') as f:
    #html_content_1 = f.read()


# Display HTML content in Streamlit
#st.markdown(html_content_1, unsafe_allow_html=True)