"""Example usage of the Multi-modal PDF RAG system."""
import os
from pdf_processor import MultimodalPDFProcessor
from vector_store import MultimodalVectorStore
from agents import MultiAgentRAG
import config


def example_usage():
    """Example of how to use the RAG system programmatically."""
    
    # Step 1: Process PDF files
    print("Step 1: Processing PDF files...")
    processor = MultimodalPDFProcessor(processing_mode=config.PDF_PROCESSING_MODE)
    
    # Replace with your PDF path
    pdf_path = "example.pdf"  # Change this to your PDF file path
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print("Please provide a valid PDF file path.")
        return
    
    chunks = processor.process_pdf(pdf_path)
    print(f"Processed {len(chunks)} chunks from PDF")
    
    # Step 2: Create vector store and add documents
    print("\nStep 2: Creating vector store...")
    vector_store = MultimodalVectorStore()
    vector_store.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to vector store")
    
    # Step 3: Create multi-agent RAG system
    print("\nStep 3: Initializing multi-agent RAG system...")
    rag_system = MultiAgentRAG(vector_store)
    print("RAG system initialized!")
    
    # Step 4: Query the system
    print("\nStep 4: Querying the system...")
    questions = [
        "What is this document about?",
        "Summarize the main points.",
        "What are the key findings?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("Thinking...")
        answer = rag_system.query(question)
        print(f"Answer: {answer}")
        print("-" * 80)


if __name__ == "__main__":
    # Check if API key is set
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("ERROR: Please set your OPENAI_API_KEY in the .env file")
        print("Copy .env.example to .env and add your API key")
    else:
        example_usage()

