# 📚 Google Colab Setup Guide

This guide will help you run the Multi-modal PDF RAG system on Google Colab.

## Quick Start

1. **Open the Colab Notebook**: `Multi_Modal_RAG_Colab.ipynb`
2. **Set your API keys** in Step 2
3. **Upload project files** in Step 3
4. **Run all cells** in order

## Detailed Steps

### Step 1: Get API Keys

#### Hugging Face API Key (Required)
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "colab-rag")
4. Select "Read" access
5. Copy the token

#### Tavily API Key (Optional - for web search)
1. Go to https://tavily.com/
2. Sign up for free
3. Get your API key from the dashboard

### Step 2: Install Dependencies

Run the first cell to install all required packages. This includes:
- LangChain and LangGraph
- PDF processing libraries
- Vector store (ChromaDB)
- Sentence transformers for embeddings
- Image processing tools

### Step 3: Configure API Keys

Replace `YOUR_HUGGINGFACE_API_KEY_HERE` with your actual Hugging Face API key.

### Step 4: Upload Project Files

You have two options:

**Option A: Upload individual files**
- Upload all `.py` files from the project:
  - `config.py`
  - `pdf_processor.py`
  - `vector_store.py`
  - `agents.py`
  - `huggingface_fallback.py`
  - `image_embeddings.py`

**Option B: Upload as ZIP**
- Create a ZIP file with all project files
- Upload the ZIP file
- The notebook will extract it automatically

### Step 5: Initialize System

Run the initialization cell. This will:
- Create the vector store with Hugging Face embeddings
- Initialize the RAG system with Hugging Face LLM

### Step 6: Upload and Process PDF

1. Upload your PDF file(s)
2. The system will extract text, images, and tables
3. Documents will be added to the vector store

### Step 7: Ask Questions

Ask questions about your documents! For example:
- "tell me about the image in doc"
- "What is the main topic?"
- "Summarize the document"

## Troubleshooting

### Hugging Face API Returns None

**Problem**: The API returns `None` or errors.

**Solutions**:
1. **Check your API key**: Make sure it's correct and has read access
2. **Wait for model loading**: Some models take 30-60 seconds to load on first use
3. **Try a different model**: The system will automatically try fallback models:
   - `google/flan-t5-base` (default)
   - `microsoft/DialoGPT-medium`
   - `gpt2` (fallback)

4. **Check model availability**: Some models may not be available via the Inference API. Try:
   ```python
   # In a cell, test the API directly:
   from huggingface_fallback import HuggingFaceLLM
   import os
   
   hf_llm = HuggingFaceLLM(
       api_key=os.environ["HUGGINGFACE_API_KEY"],
       model="google/flan-t5-base"
   )
   result = hf_llm.generate("Hello", max_length=50)
   print(result)
   ```

### Out of Memory

**Problem**: Colab runs out of memory.

**Solutions**:
1. **Restart runtime**: Runtime > Restart runtime
2. **Use smaller models**: Change model in config
3. **Process PDFs one at a time**: Don't process multiple large PDFs at once
4. **Upgrade to Colab Pro**: For more memory

### PDF Processing Fails

**Problem**: PDFs don't process correctly.

**Solutions**:
1. **Check dependencies**: Make sure poppler-utils and tesseract-ocr are installed
2. **Try simpler PDF**: Test with a simple text-only PDF first
3. **Check PDF format**: Some encrypted or corrupted PDFs may not work

### Import Errors

**Problem**: Can't import modules.

**Solutions**:
1. **Check file upload**: Make sure all `.py` files are uploaded
2. **Check file names**: File names must match exactly (case-sensitive)
3. **Restart runtime**: Sometimes helps with import issues

## Model Recommendations

### Text Generation (LLM)
- **Best for general use**: `google/flan-t5-base` (fast, reliable)
- **Better quality**: `microsoft/DialoGPT-medium` (conversational)
- **Fallback**: `gpt2` (always available)

### Image Understanding
- **Default**: `Salesforce/blip-image-captioning-large` (good quality)
- **Faster**: `Salesforce/blip-image-captioning-base` (faster, lower quality)

### Embeddings
- **Default**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API needed)

## Advanced Usage

### Change Models

Edit the config in Step 2:
```python
os.environ["HUGGINGFACE_LLM_MODEL"] = "microsoft/DialoGPT-medium"
os.environ["HUGGINGFACE_MULTIMODAL_MODEL"] = "Salesforce/blip-image-captioning-base"
```

### Process Multiple PDFs

```python
# Upload multiple PDFs
uploaded = files.upload()

# Process all
for filename in uploaded.keys():
    if filename.endswith('.pdf'):
        chunks = processor.process_pdf(filename)
        all_chunks.extend(chunks)

# Add all at once
vector_store.add_documents(all_chunks)
```

### Interactive Chat

Use the interactive chat cell to have a conversation with your documents:
- Type questions
- Get answers
- Type 'quit' to exit

## Tips

1. **Save your work**: Download the notebook regularly
2. **Use GPU**: Enable GPU runtime for faster processing (Runtime > Change runtime type > GPU)
3. **Monitor usage**: Check Colab usage limits
4. **Test incrementally**: Test with small PDFs first

## Support

If you encounter issues:
1. Check the error messages carefully
2. Try the troubleshooting steps above
3. Check Hugging Face model pages for availability
4. Verify API keys are correct

## Next Steps

- Process more PDFs
- Ask complex questions
- Experiment with different models
- Build your own RAG system!

