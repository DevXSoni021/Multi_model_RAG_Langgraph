"""Image embedding module using CLIP for semantic image search."""
import base64
from io import BytesIO
from typing import List, Optional
from PIL import Image
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")


class ImageEmbedder:
    """Image embedding using CLIP model."""
    
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initialize the image embedder.
        
        Args:
            model_name: CLIP model name (default: clip-ViT-B-32)
        """
        self.model = None
        self.model_name = model_name
        if CLIP_AVAILABLE:
            try:
                # Load CLIP model for image embeddings
                self.model = SentenceTransformer(model_name)
                print(f"✓ Loaded CLIP model: {model_name}")
            except Exception as e:
                print(f"Warning: Could not load CLIP model: {e}")
                print("Falling back to text-only embeddings")
        else:
            print("Warning: CLIP not available, image embeddings disabled")
    
    def embed_image(self, image_base64: str) -> Optional[List[float]]:
        """
        Generate embedding for a base64-encoded image.
        
        Args:
            image_base64: Base64-encoded image string
            
        Returns:
            Image embedding vector or None if embedding fails
        """
        if not self.model or not image_base64:
            return None
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate embedding
            embedding = self.model.encode(image, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Warning: Could not embed image: {e}")
            return None
    
    def embed_images(self, images_base64: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple images.
        
        Args:
            images_base64: List of base64-encoded image strings
            
        Returns:
            List of image embedding vectors
        """
        if not self.model:
            return [None] * len(images_base64)
        
        embeddings = []
        for img_base64 in images_base64:
            embedding = self.embed_image(img_base64)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using CLIP (for text-to-image search).
        
        Args:
            text: Text query
            
        Returns:
            Text embedding vector or None if embedding fails
        """
        if not self.model:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Warning: Could not embed text: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if CLIP model is available."""
        return self.model is not None

