import streamlit as st
from rag import retriever, bm25, llm, texts

st.title("RAG Document Assistant")

query = st.text_input("Ask a question")

if query:

    docs = retriever.invoke(query)

    query_tokens = query.split()
    bm25_results = bm25.get_top_n(query_tokens, texts, n=3)

    context_docs = docs + bm25_results

    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)