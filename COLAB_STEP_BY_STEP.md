# 📚 Complete Google Colab Step-by-Step Guide

## Quick Setup (5 Steps)

### Step 1: Enable GPU
1. Go to **Runtime** → **Change runtime type**
2. Select **GPU** (T4 or better)
3. Click **Save**

### Step 2: Install Dependencies
```python
!pip install -q langchain>=0.1.0 langchain-openai>=0.0.2 langchain-community>=0.0.10 langgraph>=0.0.20
!pip install -q unstructured[pdf] pypdf pdf2image Pillow
!pip install -q chromadb faiss-cpu
!pip install -q sentence-transformers torch torchvision
!pip install -q duckduckgo-search tavily-python
!pip install -q python-dotenv requests opencv-python
!pip install -q numpy==1.24.3 pydantic>=2.7.4,<3.0.0

!apt-get update -qq
!apt-get install -y -qq poppler-utils tesseract-ocr

print("✅ All dependencies installed!")
```

### Step 3: Set API Keys
```python
import os

# ⚠️ REPLACE WITH YOUR ACTUAL API KEYS ⚠️
HUGGINGFACE_API_KEY = "YOUR_HUGGINGFACE_API_KEY_HERE"
TAVILY_API_KEY = "YOUR_TAVILY_API_KEY_HERE"  # Optional

os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["USE_HUGGINGFACE_PRIMARY"] = "true"
os.environ["USE_OPENAI_EMBEDDINGS"] = "false"
os.environ["USE_OPENAI_FALLBACK"] = "false"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

print("✅ API keys configured!")
```

### Step 4: Upload Project Files
**Option A: Upload from GitHub (if you have the repo)**
```python
!git clone https://github.com/yourusername/Multi-modal-agent-pdf-RAG-with-langgraph.git
%cd Multi-modal-agent-pdf-RAG-with-langgraph
```

**Option B: Upload files manually**
1. Upload all `.py` files:
   - `config.py`
   - `pdf_processor.py`
   - `vector_store.py`
   - `agents.py`
   - `huggingface_fallback.py`
   - `image_embeddings.py`

### Step 5: Initialize and Use
```python
from pdf_processor import MultimodalPDFProcessor
from vector_store import MultimodalVectorStore
from agents import MultiAgentRAG
import config

# Initialize
print("📦 Initializing vector store...")
vector_store = MultimodalVectorStore()
print(f"✅ Vector store initialized: {vector_store.embedding_type}")

print("🤖 Initializing RAG system...")
rag_system = MultiAgentRAG(vector_store, use_huggingface_primary=True)
print(f"✅ RAG system initialized: {rag_system.primary_llm_type}")

print("\n🎉 System ready!")
```

### Step 6: Upload and Process PDF
```python
from google.colab import files

# Upload PDF
uploaded = files.upload()

# Process
processor = MultimodalPDFProcessor(processing_mode=config.PDF_PROCESSING_MODE)
all_chunks = []

for filename in uploaded.keys():
    if filename.endswith('.pdf'):
        print(f"\n📄 Processing {filename}...")
        chunks = processor.process_pdf(filename)
        all_chunks.extend(chunks)
        print(f"✅ Extracted {len(chunks)} chunks")

# Add to vector store
if all_chunks:
    print(f"\n💾 Adding {len(all_chunks)} chunks to vector store...")
    vector_store.add_documents(all_chunks)
    print("✅ Documents added!")
```

### Step 7: Ask Questions
```python
question = "tell me about the image in doc"

print(f"❓ Question: {question}\n")
print("🤔 Thinking...\n")

try:
    answer = rag_system.query(question)
    print(f"\n💬 Answer:\n{answer}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
```

## Complete Notebook

The complete notebook `Complete_Colab_Notebook.ipynb` has all these steps with all code included. Just:
1. Open it in Google Colab
2. Set your API keys
3. Run all cells

## Troubleshooting

### GPU Not Available
- Use CPU (slower but works)
- The system will automatically use CPU if GPU not available

### Out of Memory
- Restart runtime: Runtime → Restart runtime
- Process smaller PDFs
- Use smaller models

### Hugging Face API Returns None
- Check API key is correct
- Wait 30-60 seconds (models need to load)
- System will try fallback models automatically

## Next Steps

- Process more PDFs
- Ask complex questions
- Experiment with different models
- Build your own RAG system!

