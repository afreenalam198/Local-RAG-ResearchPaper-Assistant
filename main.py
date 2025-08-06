from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# LLM Model
llm_model = OllamaLLM(model="llama3.2")

# RAG Prompt
template = """ 
You are a helpful research assistant specialized in academic papers, particularly in the field of Healthcare AI.

Below is extracted content from one or more research papers related to healthcare and artificial intelligence. Use only the information provided in this context to answer the user's question. Do not make up facts.
{context}

Now, answer the following question clearly and concisely:
{question}

If the answer is not stated or supported by the context, say: "The answer is not available in the provided papers."

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm_model

while True:
    print("\n\n--------------------------------------")
    question = input("Ask me anything about your research paper(s) (Type q to quit): ")
    print("\n\n")
    if question == "q":
        print("Exiting...")
        break

    context = retriever.invoke(question)
    result = chain.invoke({"context": context, "question": question})
    print(result)

    docs = retriever.invoke(question)
    
    # print("\n🔍 Retrieved Chunks:")
    # for i, doc in enumerate(docs):
    #     print(f"\n--- Chunk {i+1} ---")
    #     print(doc.page_content[:1000])  # Truncate for readability