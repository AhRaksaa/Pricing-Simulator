import streamlit as st

with open('style/logo.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    
    st.markdown('<h1 style="color: #205527;">Embedded Power BI Dashboard</h1>', unsafe_allow_html=True)

    # Embedding Power BI report in an iframe
    st.write("### Power BI Dashboard:")

if __name__ == "__main__":
    main()
