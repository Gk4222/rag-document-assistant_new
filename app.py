import streamlit as st
@st.cache_resource
def load_rag():
    from rag import retriever, bm25, llm, texts
    return retriever, bm25, llm, texts

retriever, bm25, llm, texts = load_rag()

st.title("📄 RAG Document Assistant")
st.write("Ask questions about the uploaded document.")

query = st.text_input("Ask a question")

if query:
    with st.spinner("Searching document and generating answer..."):
        from rag import retriever, bm25, llm, texts

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