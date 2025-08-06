import streamlit as st
from vector import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Load LLM
llm_model = OllamaLLM(model="llama3.2")

# RAG Prompt Template
template = """ 
You are a helpful research assistant specialized in academic papers, particularly in the field of Healthcare AI.

Below is extracted content from one or more research papers related to healthcare and artificial intelligence. Use only the information provided in this context to answer the user's question. Do not make up facts.

---
{context}
---

Now, answer the following question clearly and concisely:
{question}

If the answer is not stated or supported by the context, say: "The answer is not available in the provided papers."
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm_model

# Streamlit UI
st.set_page_config(page_title="ResearchPal Chat", layout="wide")
st.title("ResearchPal - Local Paper Assistant")

st.markdown("Ask research questions about your uploaded Healthcare AI papers.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_question = st.text_input("Ask your question:", key="question_input")

if st.button("Ask") and user_question:
    with st.spinner("Searching papers..."):

        docs = retriever.invoke(user_question)
        context = "\n\n".join([doc.page_content for doc in docs])

        result = chain.invoke({"context": context, "question": user_question})

        st.session_state.chat_history.append((user_question, result))

# Display chat history
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**ResearchPal:** {a}")
    st.markdown("---")

# Debug view
with st.expander("Show Retrieved Chunks (Debugging)"):
    if user_question:
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.code(doc.page_content[:1000])  # Truncated for display
