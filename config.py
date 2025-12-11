"""Configuration settings for the Multi-modal RAG system."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# LangSmith Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multimodal-rag")

# Model Configuration
LLM_MODEL = "gpt-4-1106-preview"  # Can switch to "gpt-3.5-turbo" for lower rate limits
VISION_MODEL = "gpt-4-vision-preview"
EMBEDDING_MODEL = "text-embedding-3-large"

# Hugging Face Configuration
# Best models for text generation (LLM):
# - "google/flan-t5-base" (fast, good for general tasks)
# - "google/flan-t5-large" (better quality)
# - "microsoft/DialoGPT-medium" (good for conversations)
# - "meta-llama/Llama-2-7b-chat-hf" (requires access, better quality)
# - "mistralai/Mistral-7B-Instruct-v0.2" (requires access, high quality)
# Use a smaller, faster model for local inference (no API needed)
HUGGINGFACE_LLM_MODEL = os.getenv("HUGGINGFACE_LLM_MODEL", "distilgpt2")  # Changed to distilgpt2 for faster local loading

# Best models for image understanding:
# - "Salesforce/blip-image-captioning-base" (fast, good for descriptions)
# - "Salesforce/blip-image-captioning-large" (better quality)
# - "nlpconnect/vit-gpt2-image-captioning" (good balance)
# - "microsoft/git-base" (for detailed descriptions)
HUGGINGFACE_MULTIMODAL_MODEL = "Salesforce/blip-image-captioning-large"  # Good for image understanding
USE_HUGGINGFACE_FALLBACK = os.getenv("USE_HUGGINGFACE_FALLBACK", "false").lower() == "true"
USE_HUGGINGFACE_PRIMARY = os.getenv("USE_HUGGINGFACE_PRIMARY", "true").lower() == "true"  # Use HF as primary if API key available

# Rate Limit Configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

# Vector Store Configuration
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
CHROMA_COLLECTION_NAME = "multimodal_pdf_rag"
MAX_RETRIEVAL_DOCS = 3  # Limit retrieved docs when images are present
MAX_IMAGES_PER_QUERY = 2  # Maximum images to include in a single query

# PDF Processing Configuration
PDF_PROCESSING_MODE = "hi_res"  # Options: fast, ocr_only, hi_res
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Agent Configuration
MAX_ITERATIONS = 30  # Increased to handle complex queries
TEMPERATURE = 0.0

