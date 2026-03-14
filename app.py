import streamlit as st
import os
from langchain_groq import ChatGroq

st.title("Groq Test")

groq_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_key
)

query = st.text_input("Ask something")

if query:
    with st.spinner("Generating..."):
        response = llm.invoke(query)

    st.write(response.content)