"""Streamlit application for Multi-modal PDF RAG with LangGraph."""
import streamlit as st
import os
from pathlib import Path
import tempfile

# Fix for protobuf compatibility with chromadb
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Ensure Poppler and Tesseract are in PATH (for macOS Homebrew installation)
if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

from pdf_processor import MultimodalPDFProcessor
from vector_store import MultimodalVectorStore
from agents import MultiAgentRAG
import config

# Page configuration
st.set_page_config(
    page_title="Multi-modal PDF RAG with LangGraph",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def initialize_system():
    """Initialize the RAG system."""
    try:
        if st.session_state.vector_store is None:
            # Always create vector store with Hugging Face embeddings
            st.session_state.vector_store = MultimodalVectorStore()
            # Verify we're using Hugging Face
            if hasattr(st.session_state.vector_store, 'embedding_type'):
                if st.session_state.vector_store.embedding_type != "huggingface":
                    st.warning(f"⚠️ Vector store is using {st.session_state.vector_store.embedding_type} embeddings. Expected Hugging Face.")
        
        if st.session_state.rag_system is None:
            # ALWAYS use Hugging Face as primary if API key is available (ignore USE_HUGGINGFACE_PRIMARY flag)
            use_hf_primary = bool(config.HUGGINGFACE_API_KEY)
            if use_hf_primary:
                print(f"Initializing RAG system with Hugging Face as primary (model: {config.HUGGINGFACE_LLM_MODEL})")
            st.session_state.rag_system = MultiAgentRAG(
                st.session_state.vector_store,
                use_huggingface_primary=use_hf_primary
            )
            # Verify which LLM is being used
            if hasattr(st.session_state.rag_system, 'primary_llm_type'):
                print(f"✓ RAG system initialized with primary LLM: {st.session_state.rag_system.primary_llm_type}")
                if st.session_state.rag_system.primary_llm_type != "huggingface" and use_hf_primary:
                    st.warning(f"⚠️ Expected Hugging Face but got {st.session_state.rag_system.primary_llm_type}. Check your Hugging Face API key.")
        
        return True
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False


def process_pdf_files(uploaded_files):
    """Process uploaded PDF files."""
    # PDF processing doesn't require OpenAI - it only needs unstructured library
    # Embeddings will use Hugging Face (local) or OpenAI (if configured)
    
    processor = MultimodalPDFProcessor(processing_mode=config.PDF_PROCESSING_MODE)
    
    with st.spinner("Processing PDF files..."):
        all_chunks = []
        temp_files = []
        
        try:
            for uploaded_file in uploaded_files:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_files.append(tmp_file.name)
                
                # Process PDF
                chunks = processor.process_pdf(tmp_file.name)
                all_chunks.extend(chunks)
            
            # Add to vector store
            if all_chunks:
                # Always create a fresh vector store to ensure we use the right embeddings
                # Delete old collection if it exists (might have been created with OpenAI)
                if st.session_state.vector_store is not None:
                    try:
                        st.session_state.vector_store.delete_collection()
                    except:
                        pass
                
                # Create new vector store with Hugging Face embeddings
                st.session_state.vector_store = MultimodalVectorStore()
                
                # Verify we're using Hugging Face embeddings
                if st.session_state.vector_store.embedding_type != "huggingface":
                    st.warning(f"Warning: Using {st.session_state.vector_store.embedding_type} embeddings instead of Hugging Face")
                
                st.session_state.vector_store.add_documents(all_chunks)
                st.session_state.documents_loaded = True
                
                # Reinitialize RAG system with updated vector store
                # ALWAYS use Hugging Face as primary if API key is available
                use_hf_primary = bool(config.HUGGINGFACE_API_KEY)
                if use_hf_primary:
                    print(f"Initializing RAG system with Hugging Face as primary (model: {config.HUGGINGFACE_LLM_MODEL})")
                st.session_state.rag_system = MultiAgentRAG(
                    st.session_state.vector_store,
                    use_huggingface_primary=use_hf_primary
                )
                # Verify which LLM is being used
                if hasattr(st.session_state.rag_system, 'primary_llm_type'):
                    print(f"✓ RAG system initialized with primary LLM: {st.session_state.rag_system.primary_llm_type}")
                
                st.success(f"Successfully processed {len(uploaded_files)} PDF(s) with {len(all_chunks)} chunks!")
                return True
            else:
                st.warning("No content extracted from PDF files.")
                return False
                
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            return False
        finally:
            # Clean up temporary files
            for tmp_file in temp_files:
                try:
                    os.unlink(tmp_file)
                except:
                    pass


def main():
    """Main application function."""
    st.title("📚 Multi-modal PDF RAG with LangGraph")
    st.markdown("""
    This application allows you to:
    - Upload PDF files with text, images, and tables
    - Query the documents using a multi-agent RAG system
    - Get answers from both your documents and web search
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check API keys and show status
        if config.HUGGINGFACE_API_KEY:
            st.success("✅ Hugging Face API Key configured (Primary)")
            if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your_openai_api_key_here":
                st.info("ℹ️ OpenAI API Key configured (Fallback)")
        elif config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your_openai_api_key_here":
            st.warning("⚠️ Using OpenAI (Hugging Face not configured)")
        else:
            st.error("⚠️ Please set HUGGINGFACE_API_KEY or OPENAI_API_KEY in .env file")
        
        st.divider()
        
        st.header("📄 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if st.button("Process PDFs", type="primary"):
            if uploaded_files:
                process_pdf_files(uploaded_files)
            else:
                st.warning("Please upload at least one PDF file")
        
        # Show document status
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded")
            if st.session_state.vector_store:
                try:
                    collection_size = st.session_state.vector_store.get_collection_size()
                    st.info(f"📊 Collection size: {collection_size} chunks")
                except:
                    pass
        else:
            st.info("📝 No documents loaded yet")
        
        st.divider()
        
        # Clear button
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.session_state.vector_store:
                try:
                    st.session_state.vector_store.delete_collection()
                    st.session_state.vector_store = None
                    st.session_state.rag_system = None
                    st.session_state.documents_loaded = False
                    st.session_state.chat_history = []
                    st.success("Data cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing data: {str(e)}")
    
    # Main content area
    if not initialize_system():
        st.stop()
    
    # Chat interface
    st.header("💬 Chat with your documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if not st.session_state.documents_loaded:
                        response = "Please upload and process PDF documents first before asking questions."
                    else:
                        # Show progress for rate limit handling
                        status_placeholder = st.empty()
                        response = None
                        
                        try:
                            response = st.session_state.rag_system.query(prompt)
                        except Exception as query_error:
                            error_str = str(query_error).lower()
                            if "rate limit" in error_str or "429" in error_str:
                                status_placeholder.warning("⚠️ Rate limit reached. Retrying with delay...")
                                import time
                                time.sleep(3)
                                response = st.session_state.rag_system.query(prompt)
                            else:
                                raise query_error
                        
                        if response:
                            st.markdown(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                        else:
                            st.warning("No response generated. Please try again.")
                            st.session_state.chat_history.append({"role": "assistant", "content": "No response generated."})
                            
                except Exception as e:
                    error_msg = str(e)
                    # Provide helpful message for rate limits
                    if "rate limit" in error_msg.lower() or "429" in error_msg or "tpm" in error_msg.lower():
                        error_msg = "⚠️ OpenAI API rate limit reached. Please wait a moment and try again. The system will automatically retry."
                    else:
                        error_msg = f"Error: {error_msg[:500]}"
                    
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()

