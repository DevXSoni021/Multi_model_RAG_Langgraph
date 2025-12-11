# Project Summary: Multi-modal PDF RAG with LangGraph

## ✅ Project Complete!

This project implements a complete **Multi-modal PDF RAG (Retrieval-Augmented Generation) system** using **LangGraph** for multi-agent orchestration.

## 📁 Project Structure

```
Multi-modal-agent-pdf-RAG-with-langgraph/
├── app.py                 # Streamlit web application
├── main.py                # Application entry point
├── config.py              # Configuration and settings
├── pdf_processor.py       # PDF processing with multimodal support
├── vector_store.py        # Vector database for embeddings
├── agents.py              # LangGraph multi-agent system
├── example_usage.py       # Example script for programmatic usage
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── README_SETUP.md       # Detailed setup instructions
├── QUICKSTART.md         # Quick start guide
├── readme.md             # Original project readme
└── imgs/                 # Project images
    ├── BertAndGPT.jpg
    ├── LangGraph.png
    ├── Lang_Smith.png
    ├── LangSmith.png
    ├── multi_model_vector_retriever.png
    └── RAG.png
```

## 🎯 Key Features

### 1. **Multi-modal PDF Processing** (`pdf_processor.py`)
- Extracts text, images, and tables from PDFs
- Uses `unstructured` library for high-quality extraction
- Supports multiple processing modes (fast, OCR, hi-res)
- Encodes images as base64 for storage

### 2. **Vector Store** (`vector_store.py`)
- Uses ChromaDB for persistent vector storage
- OpenAI embeddings for semantic search
- Stores multimodal content (text + images)
- Retrieval with metadata filtering

### 3. **Multi-Agent System** (`agents.py`)
- **Supervisor Agent**: Orchestrates the workflow
- **Retriever Agent**: Searches PDF documents
- **Web Search Agent**: Searches the web for current information
- Uses LangGraph for stateful multi-agent coordination
- Intelligent routing based on query type

### 4. **Streamlit UI** (`app.py`)
- User-friendly web interface
- PDF upload and processing
- Interactive chat interface
- Document management
- Real-time query processing

## 🔧 Technology Stack

- **LangChain**: Framework for LLM applications
- **LangGraph**: Multi-agent orchestration
- **OpenAI**: LLM and embeddings
- **ChromaDB**: Vector database
- **Unstructured**: PDF processing
- **Streamlit**: Web interface
- **Tavily/DuckDuckGo**: Web search

## 🚀 Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Architecture

```
User Query
    ↓
Supervisor Agent (LangGraph)
    ↓
    ├─→ Retriever Agent → PDF Vector Store
    │
    └─→ Web Search Agent → Internet
    ↓
Combined Response
```

## 🎓 How It Works

1. **PDF Processing**: PDFs are processed to extract text, images, and tables
2. **Embedding**: Content is embedded and stored in a vector database
3. **Query Routing**: Supervisor agent decides which agent to use:
   - **Retriever**: For questions about uploaded PDFs
   - **Web Search**: For current information not in PDFs
4. **Response Generation**: Selected agent processes the query and returns results
5. **Iteration**: Process continues until supervisor decides to finish

## 🔑 Configuration

Key settings in `config.py`:
- `LLM_MODEL`: GPT model for agents (default: "gpt-4-1106-preview")
- `VISION_MODEL`: For image processing (default: "gpt-4-vision-preview")
- `EMBEDDING_MODEL`: For embeddings (default: "text-embedding-3-large")
- `CHUNK_SIZE`: Text chunk size (default: 1000)
- `MAX_ITERATIONS`: Max agent iterations (default: 15)

## 📝 Usage Examples

### Via Streamlit UI
1. Upload PDF files
2. Process them
3. Ask questions in the chat

### Programmatically
```python
from pdf_processor import MultimodalPDFProcessor
from vector_store import MultimodalVectorStore
from agents import MultiAgentRAG

# Process PDF
processor = MultimodalPDFProcessor()
chunks = processor.process_pdf("document.pdf")

# Create vector store
vector_store = MultimodalVectorStore()
vector_store.add_documents(chunks)

# Create RAG system
rag = MultiAgentRAG(vector_store)

# Query
answer = rag.query("What is this document about?")
```

## 🐛 Troubleshooting

- **API Key Issues**: Ensure `.env` file has correct `OPENAI_API_KEY`
- **PDF Processing**: Check PDF is not corrupted and has sufficient disk space
- **Import Errors**: Run `pip install -r requirements.txt`
- **Web Search**: Install `duckduckgo-search` or set `TAVILY_API_KEY`

## 📚 Documentation

- `README_SETUP.md`: Detailed setup instructions
- `QUICKSTART.md`: Quick start guide
- `example_usage.py`: Code examples
- `readme.md`: Original project documentation

## 🎉 Next Steps

- Customize models and parameters in `config.py`
- Add more agents or tools
- Deploy to production (Streamlit Cloud, AWS, etc.)
- Add monitoring with LangSmith
- Enhance image processing capabilities

## 📄 License

This project is based on the original work by Wei Zhang. See `readme.md` for contact information.

---

**Built with ❤️ using LangGraph, LangChain, and OpenAI**

