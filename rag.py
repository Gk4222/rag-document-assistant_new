from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
import os
import streamlit as st

groq_key=os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
# -----------------------------
# 1️⃣ Load PDF
# -----------------------------

loader = PyPDFLoader("data/document.pdf")
documents = loader.load()

# -----------------------------
# 2️⃣ Split text into chunks
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = splitter.split_documents(documents)

# -----------------------------
# 3️⃣ Create BM25 index
# -----------------------------

corpus = [doc.page_content.split() for doc in texts]
bm25 = BM25Okapi(corpus)

# -----------------------------
# 4️⃣ Create embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 5️⃣ Create FAISS vector database
# -----------------------------

vectorstore = FAISS.from_documents(texts, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# -----------------------------
# 6️⃣ Initialize LLM
# -----------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_key
)
# -----------------------------
# 7️⃣ Ask question
# -----------------------------

query = input("\nAsk a question: ")

# -----------------------------
# 8️⃣ FAISS Retrieval
# -----------------------------

docs = retriever.invoke(query)

print("\n========== FAISS Retrieved Context ==========\n")

for i, doc in enumerate(docs):
    print(f"Chunk {i+1}:\n{doc.page_content}")
    print("\n---------------------------------------------\n")

# -----------------------------
# 9️⃣ BM25 Retrieval
# -----------------------------

query_tokens = query.split()

bm25_results = bm25.get_top_n(query_tokens, texts, n=3)

print("\n========== BM25 Retrieved Context ==========\n")

for i, doc in enumerate(bm25_results):
    print(f"Chunk {i+1}:\n{doc.page_content}")
    print("\n---------------------------------------------\n")

# -----------------------------
# 🔟 Retrieval Comparison
# -----------------------------

print("\nRetrieval Comparison:")
print("FAISS chunks:", len(docs))
print("BM25 chunks:", len(bm25_results))

# -----------------------------
# 11️⃣ Combine context
# -----------------------------

context_docs = docs + bm25_results

context = "\n\n".join([doc.page_content for doc in context_docs])

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

# -----------------------------
# 12️⃣ Generate Answer
# -----------------------------

response = llm.invoke(prompt)

print("\n========== Final Answer ==========\n")
print(response.content)