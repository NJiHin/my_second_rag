import os
import glob
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Configuration
RESOURCES_FOLDER = "Resources"
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20

def initialize_embeddings():
    """Initialize HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def initialize_vector_store(embeddings):
    """Initialize ChromaDB vector store."""
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    """
    Extract text from PDF using PyMuPDF with enhanced text extraction.
    Args:
        pdf_path: Path to the PDF file
    Returns:
        List of Document objects with extracted text
    """
    try:
        print(f"Processing: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        
        # Add metadata to documents
        for i, page in enumerate(pages):
            page.metadata.update({
                "source_file": os.path.basename(pdf_path),
                "page_number": i + 1,
                "total_pages": len(pages)
            })
        
        print(f"Successfully extracted {len(pages)} pages from {os.path.basename(pdf_path)}")
        return pages # langchain Document object
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return []

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Chunk documents using TokenTextSplitter.
    Args:
        documents: List of Document objects to chunk
    Returns:
        List of chunked Document objects
    """
    text_splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunked_docs = text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunked_docs):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs

def process_pdf_collection(pdf_folder: str = RESOURCES_FOLDER) -> List[Document]:
    """
    Process all PDF files in a folder.
    Args:
        pdf_folder: Path to folder containing PDF files
    Returns:
        List of all processed documents
    """
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    all_documents = []
    for pdf_file in pdf_files:
        documents = extract_text_from_pdf(pdf_file)
        all_documents.extend(documents)
    
    return all_documents

def save_to_vector_store(documents: List[Document], vector_store: Chroma) -> int:
    """
    Save documents to ChromaDB vector store.
    Args:
        documents: List of Document objects to save
        vector_store: ChromaDB vector store instance
    Returns:
        Number of documents successfully saved
    """
    if not documents:
        print("No documents to save")
        return 0
    
    try:
        # Add documents to vector store
        vector_store.add_documents(documents)
        
        # Persist the database
        vector_store.persist()
        
        print(f"Successfully saved {len(documents)} chunks to ChromaDB")
        return len(documents)
        
    except Exception as e:
        print(f"Error saving to vector store: {str(e)}")
        return 0

def data_ingestion(pdf_path: Optional[str] = None) -> dict:
    """
    Complete data ingestion pipeline using PyMuPDF and ChromaDB.
    Args:
        pdf_path: Optional specific PDF file path. If None, processes all PDFs in Resources folder.
    Returns:
        Dictionary with ingestion results and statistics
    """
    print("Starting data ingestion pipeline...")
    
    # Initialize components
    embeddings = initialize_embeddings()
    vector_store = initialize_vector_store(embeddings)
    
    # Extract documents
    if pdf_path:
        documents = extract_text_from_pdf(pdf_path)
    else:
        documents = process_pdf_collection()
    
    if not documents:
        return {"status": "error", "message": "No documents processed", "documents_count": 0, "chunks_count": 0}
    
    # Chunk documents
    chunked_documents = chunk_documents(documents)
    
    # Save to vector store
    saved_count = save_to_vector_store(chunked_documents, vector_store)
    
    # Return results
    results = {
        "status": "success",
        "documents_count": len(documents),
        "chunks_count": len(chunked_documents),
        "saved_count": saved_count,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }
    
    print("\n" + "="*50)
    print("INGESTION COMPLETE")
    print("="*50)
    for key, value in results.items():
        print(f"{key.title()}: {value}")
    
    return results

def main():
    """Main function to run the data ingestion pipeline."""
    if __name__ == "__main__":
        # Create directories if they don't exist
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        
        # Run ingestion pipeline
        results = data_ingestion()
        
        if results["status"] == "success":
            print(f"\nData ingestion completed successfully!")
            print(f"Vector database saved to: {CHROMA_DB_PATH}")
        else:
            print(f"\nData ingestion failed: {results['message']}")

if __name__ == "__main__":
    main()