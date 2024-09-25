import streamlit as st

with open('style/logo.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    
    st.markdown('<h1 style="color: #205527;">Embedded Power BI Dashboard</h1>', unsafe_allow_html=True)

    # Define the embedded link
    embedded_link = "https://app.powerbi.com/reportEmbed?reportId=e7695318-b683-4d14-954b-96f6080bb11f&autoAuth=true&ctid=66e853de-ece3-44dd-9d66-ee6bdf4159d4"

    # Display the embedded link using components
    st.components.v1.iframe(embedded_link, width=800, height=600)
    

if __name__ == "__main__":
    main()
