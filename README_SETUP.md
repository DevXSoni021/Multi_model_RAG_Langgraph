# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- (Optional) Tavily API key for web search
- (Optional) LangSmith API key for monitoring

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd Multi-modal-agent-pdf-RAG-with-langgraph
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_actual_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key  # Optional
   LANGCHAIN_API_KEY=your_langsmith_api_key  # Optional
   ```

## Running the Application

### Option 1: Using Streamlit directly
```bash
streamlit run app.py
```

### Option 2: Using the main script
```bash
python main.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

1. **Upload PDF files:**
   - Click "Upload PDF files" in the sidebar
   - Select one or more PDF files
   - Click "Process PDFs" to extract and index the content

2. **Ask questions:**
   - Type your question in the chat input
   - The system will automatically decide whether to:
     - Search your PDF documents (retriever agent)
     - Search the web (web search agent)
     - Or combine both

3. **View results:**
   - The system will provide answers based on the retrieved information
   - You can see the conversation history in the chat interface

## Features

- **Multi-modal PDF processing:** Extracts text, images, and tables from PDFs
- **Vector search:** Semantic search over your document content
- **Multi-agent system:** Intelligent routing between document retrieval and web search
- **LangGraph orchestration:** Stateful multi-agent workflows
- **Streamlit UI:** User-friendly interface for interaction

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set" error:**
   - Make sure you've created a `.env` file with your API key
   - Check that the key is correct and has sufficient credits

2. **PDF processing fails:**
   - Ensure the PDF files are not corrupted
   - Check that you have sufficient disk space
   - Try with a smaller PDF first

3. **Web search not working:**
   - Install `duckduckgo-search` or set up Tavily API key
   - The system will fall back to a mock search tool if neither is available

4. **Import errors:**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check that you're using the correct Python version (3.8+)

## Project Structure

```
Multi-modal-agent-pdf-RAG-with-langgraph/
├── app.py                 # Streamlit application
├── main.py                # Entry point
├── config.py              # Configuration settings
├── pdf_processor.py       # PDF processing module
├── vector_store.py        # Vector store implementation
├── agents.py              # Multi-agent system with LangGraph
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore file
└── imgs/                 # Project images
```

## Next Steps

- Customize the models in `config.py`
- Adjust chunk sizes for your use case
- Add more agents or tools as needed
- Deploy to production using Streamlit Cloud or other platforms

