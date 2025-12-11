"""PDF processing module for multimodal content extraction."""
import os
from typing import List, Dict, Any
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from PIL import Image
import base64
from io import BytesIO
from pypdf import PdfReader
import tempfile


class MultimodalPDFProcessor:
    """Process PDFs to extract text, images, and tables."""
    
    def __init__(self, processing_mode: str = "hi_res"):
        """
        Initialize the PDF processor.
        
        Args:
            processing_mode: Processing mode for unstructured library
                - "fast": Fast processing, basic extraction
                - "ocr_only": OCR only mode
                - "hi_res": High resolution, best quality
        """
        self.processing_mode = processing_mode
    
    def _extract_images_from_pdf(self, pdf_path: str) -> Dict[int, List[str]]:
        """
        Extract images directly from PDF using pypdf.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to list of base64-encoded images
        """
        images_by_page = {}
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                page_images = []
                if '/XObject' in page.get('/Resources', {}):
                    xobjects = page['/Resources']['/XObject'].get_object()
                    for obj_name in xobjects:
                        obj = xobjects[obj_name]
                        if obj.get('/Subtype') == '/Image':
                            try:
                                # Extract image data
                                if '/Filter' in obj:
                                    filter_type = obj['/Filter']
                                    if isinstance(filter_type, list):
                                        filter_type = filter_type[0]
                                    
                                    # Get image data
                                    img_data = obj.get_data()
                                    
                                    # Convert to base64
                                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                                    page_images.append(img_base64)
                            except Exception as e:
                                print(f"Warning: Could not extract image from page {page_num}: {e}")
                                continue
                
                if page_images:
                    images_by_page[page_num + 1] = page_images  # Page numbers are 1-indexed
        except Exception as e:
            print(f"Warning: Could not extract images using pypdf: {e}")
        
        return images_by_page
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a PDF file and extract multimodal content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing chunks with text, images, and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # First, extract images directly from PDF
        pdf_images = self._extract_images_from_pdf(pdf_path)
        print(f"Extracted {sum(len(imgs) for imgs in pdf_images.values())} images from PDF using pypdf")
        
        # Try to extract elements from PDF with error handling
        try:
            elements = partition_pdf(
                filename=pdf_path,
                strategy=self.processing_mode,
                infer_table_structure=True,
                extract_images_in_pdf=True,
                include_page_breaks=True,
            )
        except Exception as e:
            error_msg = str(e).lower()
            # Handle different types of errors
            if "tesseract" in error_msg or "ocr" in error_msg:
                # If Tesseract/OCR fails, try without OCR (use fast mode)
                try:
                    elements = partition_pdf(
                        filename=pdf_path,
                        strategy="fast",  # Fast mode doesn't require OCR
                        infer_table_structure=True,
                        extract_images_in_pdf=False,
                        include_page_breaks=True,
                    )
                except Exception as e2:
                    # Last resort: minimal processing
                    elements = partition_pdf(
                        filename=pdf_path,
                        strategy="fast",
                        infer_table_structure=False,
                        extract_images_in_pdf=False,
                        include_page_breaks=True,
                    )
            elif "poppler" in error_msg or "page count" in error_msg:
                # If image extraction fails (poppler issue), try without image extraction
                try:
                    elements = partition_pdf(
                        filename=pdf_path,
                        strategy=self.processing_mode,
                        infer_table_structure=True,
                        extract_images_in_pdf=False,  # Disable image extraction
                        include_page_breaks=True,
                    )
                except Exception as e2:
                    # Last resort: use fast mode without images or tables
                    elements = partition_pdf(
                        filename=pdf_path,
                        strategy="fast",
                        infer_table_structure=False,
                        extract_images_in_pdf=False,
                        include_page_breaks=True,
                    )
            else:
                # For other errors, try fast mode as fallback
                try:
                    elements = partition_pdf(
                        filename=pdf_path,
                        strategy="fast",
                        infer_table_structure=False,
                        extract_images_in_pdf=False,
                        include_page_breaks=True,
                    )
                except Exception as e2:
                    raise Exception(f"Failed to process PDF: {str(e)}. Fallback also failed: {str(e2)}")
        
        # First, extract all images from elements before chunking
        extracted_images = {}
        for element in elements:
            # Check for images in element metadata
            if hasattr(element, "metadata") and element.metadata:
                # Check for image_path
                if hasattr(element.metadata, "image_path") and element.metadata.image_path:
                    img_path = element.metadata.image_path
                    if os.path.exists(img_path):
                        try:
                            img_base64 = self._encode_image(img_path)
                            page_num = getattr(element.metadata, "page_number", 0)
                            extracted_images[page_num] = extracted_images.get(page_num, []) + [img_base64]
                        except Exception as e:
                            print(f"Warning: Could not encode image {img_path}: {e}")
                
                # Also check for image_base64 directly in metadata
                if hasattr(element.metadata, "image_base64") and element.metadata.image_base64:
                    page_num = getattr(element.metadata, "page_number", 0)
                    extracted_images[page_num] = extracted_images.get(page_num, []) + [element.metadata.image_base64]
            
            # Check if element itself is an image
            if hasattr(element, "image_base64") and element.image_base64:
                page_num = getattr(element.metadata, "page_number", 0) if hasattr(element, "metadata") else 0
                extracted_images[page_num] = extracted_images.get(page_num, []) + [element.image_base64]
        
        # Chunk the elements
        chunks = chunk_by_title(
            elements,
            max_characters=1000,
            combine_text_under_n_chars=200,
            new_after_n_chars=1000,
        )
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": f"chunk_{i}",
                "text": str(chunk),
                "type": chunk.__class__.__name__,
                "metadata": self._extract_metadata(chunk),
            }
            
            # Get page number for this chunk
            page_num = chunk_data["metadata"].get("page_number", 0)
            
            # Check for images in chunk metadata
            has_image = False
            image_base64 = None
            image_list = []
            
            # Priority 1: Check for images extracted by unstructured
            if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "image_path"):
                image_path = chunk.metadata.image_path
                if image_path and os.path.exists(image_path):
                    try:
                        image_base64 = self._encode_image(image_path)
                        has_image = True
                        image_list.append(image_base64)
                    except Exception:
                        pass
            
            # Priority 2: Check if there are extracted images for this page from unstructured
            if page_num in extracted_images and extracted_images[page_num]:
                for img in extracted_images[page_num]:
                    if img not in image_list:
                        image_list.append(img)
                if not has_image and image_list:
                    image_base64 = image_list[0]
                    has_image = True
            
            # Priority 3: Check for images extracted directly from PDF using pypdf
            if page_num in pdf_images and pdf_images[page_num]:
                for img in pdf_images[page_num]:
                    if img not in image_list:
                        image_list.append(img)
                if not has_image and image_list:
                    image_base64 = image_list[0]
                    has_image = True
            
            # Priority 4: Check for image_base64 in chunk metadata
            if not has_image and hasattr(chunk, "metadata") and hasattr(chunk.metadata, "image_base64"):
                if chunk.metadata.image_base64:
                    image_base64 = chunk.metadata.image_base64
                    has_image = True
                    if image_base64 not in image_list:
                        image_list.append(image_base64)
            
            chunk_data["has_image"] = has_image
            if has_image:
                # Store all images for this chunk
                chunk_data["image"] = image_base64 if image_base64 else (image_list[0] if image_list else "")
                chunk_data["images"] = image_list  # Store all images
                chunk_data["image_count"] = len(image_list)
            else:
                chunk_data["image"] = ""
                chunk_data["images"] = []
                chunk_data["image_count"] = 0
            
            processed_chunks.append(chunk_data)
        
        # If we have images but no text chunks, create image-only chunks
        all_images = {**extracted_images, **pdf_images}
        if not processed_chunks and all_images:
            for page_num, images in all_images.items():
                for img_idx, img_base64 in enumerate(images):
                    chunk_data = {
                        "id": f"image_chunk_{page_num}_{img_idx}",
                        "text": f"Image {img_idx + 1} from page {page_num}",
                        "type": "Image",
                        "metadata": {"page_number": page_num, "is_table": False},
                        "has_image": True,
                        "image": img_base64,
                        "images": [img_base64],
                        "image_count": 1
                    }
                    processed_chunks.append(chunk_data)
        
        # Also create image chunks for pages that have images but no text
        if all_images:
            pages_with_images = set(all_images.keys())
            pages_with_text = set()
            for chunk_data in processed_chunks:
                page_num = chunk_data.get("metadata", {}).get("page_number", 0)
                if page_num:
                    pages_with_text.add(page_num)
            
            pages_without_text = pages_with_images - pages_with_text
            
            for page_num in pages_without_text:
                for img_idx, img_base64 in enumerate(all_images[page_num]):
                    chunk_data = {
                        "id": f"image_only_chunk_{page_num}_{img_idx}",
                        "text": f"Image {img_idx + 1} from page {page_num}",
                        "type": "Image",
                        "metadata": {"page_number": page_num, "is_table": False},
                        "has_image": True,
                        "image": img_base64,
                        "images": [img_base64],
                        "image_count": 1
                    }
                    processed_chunks.append(chunk_data)
        
        return processed_chunks
    
    def _extract_metadata(self, element) -> Dict[str, Any]:
        """Extract metadata from an element."""
        metadata = {}
        
        if hasattr(element, "metadata"):
            elem_metadata = element.metadata
            metadata["page_number"] = getattr(elem_metadata, "page_number", None)
            metadata["filename"] = getattr(elem_metadata, "filename", None)
            
            # Extract table metadata if present
            if hasattr(elem_metadata, "table_as_cells"):
                metadata["is_table"] = True
            else:
                metadata["is_table"] = False
        
        return metadata
    
    def _encode_image(self, image_path: str, max_size_kb: int = 500) -> str:
        """
        Encode image to base64 string with compression.
        
        Args:
            image_path: Path to the image file
            max_size_kb: Maximum size in KB (default 500KB to limit tokens)
            
        Returns:
            Base64 encoded image string
        """
        try:
            from PIL import Image
            import io
            
            # Open and compress image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (max 1024px on longest side to reduce tokens)
            max_dimension = 1024
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Compress to JPEG with quality adjustment
            output = io.BytesIO()
            quality = 85
            while quality > 30:
                output.seek(0)
                output.truncate(0)
                img.save(output, format='JPEG', quality=quality, optimize=True)
                size_kb = len(output.getvalue()) / 1024
                if size_kb <= max_size_kb:
                    break
                quality -= 10
            
            # Encode to base64
            encoded_string = base64.b64encode(output.getvalue()).decode("utf-8")
            return encoded_string
        except Exception as e:
            # Fallback to original method if compression fails
            print(f"Warning: Image compression failed, using original: {e}")
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Combined list of processed chunks from all PDFs
        """
        all_chunks = []
        for pdf_path in pdf_paths:
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)
        return all_chunks

