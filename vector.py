from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
import os

embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# Ingesting the pdfs

# List containing all the pdf paths
pdf_paths = [
"./Papers/Exploring_Independent_Feature_Extraction_Techniques_for_Context-based_Image_Retrieval_in_Healthcare.pdf",
"./Papers/KDTL.pdf",
"./Papers/Multimodal_Ensemble_Fusion_Deep_Learning_Using_His.pdf"
]

# List to store all pdf contents
all_docs = []

for path in pdf_paths:
    if os.path.exists(path):
        loader = UnstructuredPDFLoader(file_path=path)
        data = loader.load()
        all_docs.extend(data)
    else:
        print("File not found: {path}")

print(f"Ingested {len(all_docs)} docs/pages/chunks from all PDFs.")

# Database Location
db_location = "./researchPal_chroma_db"
add_docs = not os.path.exists(db_location) # DB does not exist, so we will add the embeddings

if add_docs:
    print("\nSplitting documents into chunks...")

    # Split and Chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_docs)

    print(f"Total chunks created: {len(chunks)}")

    # Add to Vector Database
    vector_db = Chroma.from_documents(
        collection_name="local-rag-research-assistant",
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_location
    )
    print(f"💾 Created and saved Chroma DB at {db_location}")
else:
    vector_db = Chroma(
            collection_name="local-rag-research-assistant",
            embedding_function=embedding_model,
            persist_directory=db_location
        )
    print(f"Found existing Chroma DB at {db_location}")

# Retriever
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10}
)


