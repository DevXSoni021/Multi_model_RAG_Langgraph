# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up Environment Variables
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Optional (for enhanced features):
```bash
TAVILY_API_KEY=your_tavily_api_key_here  # For better web search
LANGCHAIN_API_KEY=your_langsmith_api_key_here  # For monitoring
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

### Step 4: Use the Application
1. Open your browser to `http://localhost:8501`
2. Upload PDF files in the sidebar
3. Click "Process PDFs"
4. Start asking questions!

## 📋 What This System Does

- **Extracts** text, images, and tables from PDFs
- **Indexes** content in a vector database for semantic search
- **Routes** queries intelligently between:
  - Document retrieval (for PDF content)
  - Web search (for current information)
- **Answers** questions using multi-agent orchestration

## 🎯 Example Questions

- "What is this document about?"
- "Summarize the key findings"
- "What are the main topics discussed?"
- "Find information about [specific topic]"
- "What does the latest research say about [topic]?" (triggers web search)

## 🔧 Troubleshooting

**Issue: "OPENAI_API_KEY not set"**
- Solution: Create a `.env` file with your API key

**Issue: PDF processing fails**
- Solution: Ensure PDFs are not corrupted and you have disk space

**Issue: Import errors**
- Solution: Run `pip install -r requirements.txt` again

## 📚 Next Steps

- Read `README_SETUP.md` for detailed setup instructions
- Check `example_usage.py` for programmatic usage
- Customize `config.py` for your needs

