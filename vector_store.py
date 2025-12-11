"""Vector store for storing and retrieving multimodal PDF chunks."""
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import config
from image_embeddings import ImageEmbedder

# Try to import Hugging Face embeddings first, fallback to OpenAI
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = False
    print("Warning: langchain_community.embeddings.HuggingFaceEmbeddings not available")

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False


class MultimodalVectorStore:
    """Vector store for multimodal PDF content."""
    
    def __init__(self, collection_name: str = None):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        # In-memory cache for images (chunk_id -> image data)
        self._image_cache = {}
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        
        # Text embeddings - Use Hugging Face as primary, OpenAI as fallback
        self.embeddings = None
        self.embedding_type = None
        
        # Try Hugging Face embeddings first (free, no API key needed for local models)
        if HUGGINGFACE_EMBEDDINGS_AVAILABLE:
            try:
                # Use a good sentence transformer model (runs locally, no API needed)
                print("Initializing Hugging Face embeddings (local model)...")
                
                # Fix for PyTorch meta tensor issue - create custom wrapper
                from sentence_transformers import SentenceTransformer
                from langchain_core.embeddings import Embeddings
                
                # Load model directly to avoid meta tensor issue
                print("Loading sentence transformer model directly...")
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
                print("✓ Model loaded successfully")
                
                # Create a custom Embeddings wrapper that uses the pre-loaded model
                class CustomHuggingFaceEmbeddings(Embeddings):
                    """Custom embeddings wrapper to avoid meta tensor issue."""
                    def __init__(self, model):
                        self.client = model
                        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    
                    def embed_documents(self, texts):
                        """Embed a list of texts."""
                        embeddings = self.client.encode(
                            texts,
                            convert_to_tensor=False,
                            normalize_embeddings=True
                        )
                        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
                    
                    def embed_query(self, text):
                        """Embed a single query text."""
                        embedding = self.client.encode(
                            text,
                            convert_to_tensor=False,
                            normalize_embeddings=True
                        )
                        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                
                # Use custom wrapper instead of HuggingFaceEmbeddings
                self.embeddings = CustomHuggingFaceEmbeddings(model)
                self.embedding_type = "huggingface"
                print("✓ Using Hugging Face embeddings (local model)")
            except Exception as e:
                print(f"Error: Could not initialize Hugging Face embeddings: {e}")
                print("This is required. Please ensure sentence-transformers is installed: pip install sentence-transformers")
                import traceback
                traceback.print_exc()
                self.embeddings = None
                raise ValueError(f"Failed to initialize Hugging Face embeddings: {e}. Please install sentence-transformers: pip install sentence-transformers")
        
        # Fallback to OpenAI embeddings ONLY if explicitly requested and Hugging Face failed
        # By default, we prefer Hugging Face to avoid quota issues
        if self.embeddings is None:
            if OPENAI_EMBEDDINGS_AVAILABLE and config.OPENAI_API_KEY:
                # Only use OpenAI if Hugging Face is not available AND user explicitly wants it
                use_openai_fallback = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
                if use_openai_fallback:
                    try:
                        print("Falling back to OpenAI embeddings (explicitly requested)...")
                        self.embeddings = OpenAIEmbeddings(
                            model=config.EMBEDDING_MODEL,
                            openai_api_key=config.OPENAI_API_KEY
                        )
                        self.embedding_type = "openai"
                        print("✓ Using OpenAI embeddings")
                    except Exception as e:
                        print(f"Warning: Could not initialize OpenAI embeddings: {e}")
                        raise ValueError("No embeddings available. Please install sentence-transformers: pip install sentence-transformers")
                else:
                    raise ValueError("Hugging Face embeddings failed. To use OpenAI embeddings, set USE_OPENAI_EMBEDDINGS=true in .env file. Otherwise, please install sentence-transformers: pip install sentence-transformers")
            else:
                raise ValueError("No embeddings available. Please install sentence-transformers: pip install sentence-transformers")
        
        # Image embeddings using CLIP
        self.image_embedder = ImageEmbedder()
        
        # Create vector store directory if it doesn't exist
        os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
        
        # Fix for Colab: Use absolute path and ensure write permissions
        vector_store_path = os.path.abspath(config.VECTOR_STORE_PATH)
        os.makedirs(vector_store_path, exist_ok=True)
        # Ensure directory is writable
        try:
            os.chmod(vector_store_path, 0o755)
        except:
            pass  # Ignore permission errors if we can't change permissions
        
        # Initialize ChromaDB client first (consistent approach)
        try:
            # Use PersistentClient for both text and image collections
            # Use absolute path to avoid readonly issues
            self.chroma_client = chromadb.PersistentClient(
                path=vector_store_path
            )
            
            # Initialize ChromaDB for text using the client
            # Try to get existing collection first, create if doesn't exist
            try:
                # Check if collection exists
                existing_collections = [col.name for col in self.chroma_client.list_collections()]
                if self.collection_name in existing_collections:
                    # Collection exists, use it
                    self.vector_store = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        client=self.chroma_client,
                    )
                    print(f"✓ Loaded existing text collection: {self.collection_name}")
                else:
                    # Create new collection
                    self.vector_store = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        client=self.chroma_client,
                    )
                    print(f"✓ Created new text collection: {self.collection_name}")
            except Exception as e:
                print(f"Warning: Error initializing text collection: {e}")
                # Fallback: try without explicit client
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=vector_store_path,
                )
                print("✓ Initialized text collection with fallback method")
        except Exception as e:
            print(f"Warning: Error creating ChromaDB client: {e}")
            # Fallback: use default Chroma initialization
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=vector_store_path,
            )
            self.chroma_client = None
            print("✓ Initialized text collection with fallback method (no explicit client)")
        
        # Initialize separate ChromaDB collection for image embeddings
        if self.image_embedder.is_available():
            try:
                if self.chroma_client is None:
                    self.chroma_client = chromadb.PersistentClient(path=vector_store_path)
                
                image_collection_name = f"{self.collection_name}_images"
                self.image_collection = self.chroma_client.get_or_create_collection(
                    name=image_collection_name,
                    metadata={"description": "Image embeddings for semantic search"}
                )
                print("✓ Image embedding collection initialized")
            except Exception as e:
                print(f"Warning: Could not initialize image collection: {e}")
                import traceback
                traceback.print_exc()
                self.image_collection = None
        else:
            self.image_collection = None
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add processed PDF chunks to the vector store.
        
        Args:
            chunks: List of processed chunks from PDF processor
        """
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Create document with text content
            doc_text = chunk.get("text", "")
            if not doc_text:
                continue
            
            # Prepare metadata - ChromaDB only accepts str, int, float, bool
            # Don't store full base64 images in metadata (too large and wrong type)
            # Store only flags and counts
            chunk_metadata = chunk.get("metadata", {})
            
            # Clean metadata to only include allowed types
            clean_metadata = {}
            for key, value in chunk_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                elif value is None:
                    continue
                else:
                    # Convert to string if not a supported type
                    clean_metadata[key] = str(value)[:100]  # Limit length
            
            metadata_dict = {
                **clean_metadata,
                "chunk_id": str(chunk.get("id", "")),
                "chunk_type": str(chunk.get("type", "")),
                "has_image": bool(chunk.get("has_image", False)),
                "image_count": int(chunk.get("image_count", 0)),
            }
            
            # Store image reference (hash only, not the actual image)
            if chunk.get("has_image"):
                metadata_dict["image_available"] = "true"
            else:
                metadata_dict["image_available"] = "false"
            
            documents.append(Document(
                page_content=doc_text,
                metadata=metadata_dict
            ))
            
            metadatas.append(clean_metadata)
            chunk_id = chunk.get("id")
            ids.append(chunk_id)
            
            # Store images in memory cache and create image embeddings
            if chunk.get("has_image"):
                images = chunk.get("images", [])
                if not images and chunk.get("image"):
                    images = [chunk.get("image")]
                if images:
                    self._image_cache[chunk_id] = {
                        "images": images,
                        "image_count": len(images)
                    }
                    
                    # Create and store image embeddings for semantic search
                    if self.image_collection and self.image_embedder.is_available():
                        for img_idx, img_base64 in enumerate(images):
                            try:
                                image_embedding = self.image_embedder.embed_image(img_base64)
                                if image_embedding is not None:
                                    # Convert numpy array to list if needed
                                    if hasattr(image_embedding, 'tolist'):
                                        embedding_list = image_embedding.tolist()
                                    elif isinstance(image_embedding, list):
                                        embedding_list = image_embedding
                                    else:
                                        embedding_list = list(image_embedding)
                                    
                                    image_id = f"{chunk_id}_img_{img_idx}"
                                    # Store image embedding with reference to chunk
                                    self.image_collection.add(
                                        ids=[image_id],
                                        embeddings=[embedding_list],
                                        metadatas=[{
                                            "chunk_id": str(chunk_id),
                                            "image_index": str(img_idx),
                                            "page_number": str(metadata_dict.get("page_number", "")),
                                            "has_text": "true" if doc_text else "false"
                                        }]
                                    )
                            except Exception as e:
                                print(f"Warning: Could not create embedding for image {img_idx} in chunk {chunk_id}: {e}")
                                continue
        
        # Add to vector store
        if not documents:
            print("Warning: No documents to add to vector store (all chunks may have been empty)")
            return
        
        try:
            # Ensure all IDs are strings
            ids = [str(id) for id in ids if id is not None]
            
            # Check that documents and ids have the same length
            if len(documents) != len(ids):
                print(f"Warning: Mismatch between documents ({len(documents)}) and ids ({len(ids)}). Adjusting...")
                # Use only the matching length
                min_len = min(len(documents), len(ids))
                documents = documents[:min_len]
                ids = ids[:min_len]
            
            self.vector_store.add_documents(documents, ids=ids)
            self.vector_store.persist()
            print(f"✓ Added {len(documents)} text chunks to vector store")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Persist image collection if available
        if self.image_collection:
            try:
                # ChromaDB persists automatically, but we can verify
                print(f"✓ Stored {len([c for c in chunks if c.get('has_image')])} image chunks with embeddings")
            except Exception as e:
                print(f"Warning: Could not persist image collection: {e}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of tuples (document, score)
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def get_retriever(self, k: int = 4):
        """
        Get a retriever for use with LangChain.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Vector store retriever
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def get_image_for_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """
        Get image data for a chunk from cache.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Dictionary with image data or None
        """
        return self._image_cache.get(chunk_id)
    
    def search_images(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Semantic search for images using CLIP embeddings.
        
        Args:
            query: Text query to search for in images
            k: Number of results to return
            
        Returns:
            List of dictionaries with image data and metadata
        """
        if not self.image_collection or not self.image_embedder.is_available():
            return []
        
        try:
            # Embed the text query using CLIP
            query_embedding = self.image_embedder.embed_text(query)
            if not query_embedding:
                return []
            
            # Search image embeddings
            results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Format results
            image_results = []
            if results.get("ids") and len(results["ids"][0]) > 0:
                for idx, image_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][idx] if results.get("metadatas") else {}
                    chunk_id = metadata.get("chunk_id", "")
                    
                    # Get actual image data from cache
                    image_data = self._image_cache.get(chunk_id)
                    if image_data:
                        image_idx = int(metadata.get("image_index", 0))
                        images = image_data.get("images", [])
                        if image_idx < len(images):
                            image_results.append({
                                "chunk_id": chunk_id,
                                "image_base64": images[image_idx],
                                "page_number": metadata.get("page_number", ""),
                                "image_index": image_idx,
                                "distance": results["distances"][0][idx] if results.get("distances") else None
                            })
            
            return image_results
        except Exception as e:
            print(f"Error searching images: {e}")
            return []
    
    def multimodal_search(self, query: str, k: int = 4, include_images: bool = True) -> Dict[str, Any]:
        """
        Perform multimodal search (text + images).
        
        Args:
            query: Search query
            k: Number of text results
            include_images: Whether to include image search results
            
        Returns:
            Dictionary with text_results and image_results
        """
        # Text search
        text_results = self.similarity_search(query, k=k)
        
        # Image search
        image_results = []
        if include_images and self.image_collection:
            image_results = self.search_images(query, k=k)
        
        return {
            "text_results": text_results,
            "image_results": image_results
        }
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.vector_store.delete_collection()
    
    def get_collection_size(self) -> int:
        """Get the number of documents in the collection."""
        return self.vector_store._collection.count()

