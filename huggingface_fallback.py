"""Hugging Face Inference API fallback for multimodal models."""
import os
import base64
import requests
from typing import Optional, Dict, Any, List
from io import BytesIO
from PIL import Image
import config

# LangChain integration
try:
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage, AIMessage
    from langchain_core.outputs import LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseLLM = object


class HuggingFaceMultimodal:
    """Hugging Face Inference API for multimodal image-text understanding."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Hugging Face multimodal model.
        
        Args:
            api_key: Hugging Face API key
            model: Model name (default: from config)
        """
        self.api_key = api_key or config.HUGGINGFACE_API_KEY
        self.model = model or config.HUGGINGFACE_MULTIMODAL_MODEL
        # Try both old and new endpoints - the API format may vary
        self.base_urls = [
            "https://api-inference.huggingface.co/models",  # Old (may be deprecated)
            "https://router.huggingface.co/models",  # New router
        ]
        
        if not self.api_key:
            print("Warning: HUGGINGFACE_API_KEY not set. Hugging Face fallback disabled.")
    
    def is_available(self) -> bool:
        """Check if Hugging Face API is available."""
        return bool(self.api_key)
    
    def describe_image(self, image_base64: str, prompt: str = "Describe this image in detail.") -> Optional[str]:
        """
        Describe an image using Hugging Face multimodal model.
        
        Args:
            image_base64: Base64-encoded image
            prompt: Text prompt/question about the image
            
        Returns:
            Description of the image or None if error
        """
        if not self.is_available():
            return None
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare request - Hugging Face Inference API format
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Convert image to bytes
            img_bytes = BytesIO()
            image.save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)
            
            # Try both endpoints
            files = {
                "image": ("image.jpg", img_bytes.getvalue(), "image/jpeg")
            }
            data = {
                "inputs": prompt if prompt else "Describe this image in detail."
            }
            
            response = None
            for base_url in self.base_urls:
                url = f"{base_url}/{self.model}"
                try:
                    response = requests.post(
                        url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=60
                    )
                    # If we get a non-410/404 response, use it
                    if response.status_code not in [410, 404]:
                        break
                except Exception as e:
                    print(f"Error with {url}: {e}")
                    continue
            
            if not response or response.status_code in [410, 404]:
                print(f"Warning: Hugging Face image-to-text API not available. Endpoints returned 410/404.")
                print("Note: The free Inference API may require Inference Endpoints (paid) for some models.")
                return None
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                # Handle different response formats
                if isinstance(result, dict):
                    return result.get("generated_text", result.get("caption", str(result)))
                elif isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        return result[0].get("generated_text", result[0].get("caption", str(result[0])))
                    return str(result[0])
                return str(result)
            elif response.status_code == 503:
                # Model is loading, wait and retry
                import time
                wait_time = 10
                print(f"Model loading, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=60
                )
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict):
                        return result.get("generated_text", result.get("caption", str(result)))
                    elif isinstance(result, list) and len(result) > 0:
                        return str(result[0])
                    return str(result)
            
            print(f"Hugging Face API error: {response.status_code} - {response.text[:200]}")
            return None
            
        except Exception as e:
            print(f"Error calling Hugging Face API: {e}")
            return None
    
    def process_images_with_text(self, images_base64: List[str], query: str) -> List[Dict[str, Any]]:
        """
        Process multiple images with a text query.
        
        Args:
            images_base64: List of base64-encoded images
            query: Text query/question
            
        Returns:
            List of descriptions for each image
        """
        results = []
        for idx, img_base64 in enumerate(images_base64):
            description = self.describe_image(img_base64, query)
            results.append({
                "image_index": idx,
                "description": description,
                "has_description": description is not None
            })
        return results


class HuggingFaceLLM:
    """Hugging Face Inference API for text generation (LLM fallback)."""
    
    def __init__(self, api_key: str = None, model: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize Hugging Face LLM.
        
        Args:
            api_key: Hugging Face API key
            model: Model name for text generation
        """
        self.api_key = api_key or config.HUGGINGFACE_API_KEY
        # Use a simpler, more reliable text generation model
        # Try multiple models in order of preference
        self.model = model or "google/flan-t5-base"  # Default model
        self.fallback_models = [
            "google/flan-t5-base",
            "microsoft/DialoGPT-medium",
            "gpt2",  # Very reliable fallback
        ]
        # Use only the router endpoint (api-inference is deprecated)
        self.base_urls = [
            "https://router.huggingface.co/models",
        ]
    
    def is_available(self) -> bool:
        """Check if Hugging Face API is available."""
        return bool(self.api_key)
    
    def generate(self, prompt: str, max_length: int = 500) -> Optional[str]:
        """
        Generate text using Hugging Face LLM.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text or None if error
        """
        if not self.is_available():
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "return_full_text": False
                }
            }
            
            # Try endpoints
            response = None
            last_error = None
            for base_url in self.base_urls:
                url = f"{base_url}/{self.model}"
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    # Check if response indicates deprecated endpoint
                    if response.status_code == 410:
                        error_text = response.text.lower()
                        if "no longer supported" in error_text or "router.huggingface.co" in error_text:
                            print(f"⚠️ Endpoint {base_url} is deprecated, trying next...")
                            continue
                    if response.status_code not in [410, 404]:
                        break
                except Exception as e:
                    last_error = e
                    print(f"Error with {url}: {e}")
                    continue
            
            if not response:
                error_msg = f"⚠️ Hugging Face API error: No response from endpoints."
                if last_error:
                    error_msg += f" Last error: {last_error}"
                print(error_msg)
                # Try fallback models
                if self.model not in self.fallback_models:
                    for fallback_model in self.fallback_models:
                        if fallback_model != self.model:
                            print(f"⚠️ Trying fallback model: {fallback_model}")
                            self.model = fallback_model
                            return self.generate(prompt, max_length)
                return None
            
            if response.status_code in [410, 404]:
                error_text = response.text[:500] if response.text else "Unknown error"
                print(f"⚠️ Hugging Face API error ({response.status_code}): {error_text}")
                # Try alternative model if current one fails
                for fallback_model in self.fallback_models:
                    if fallback_model != self.model:
                        print(f"⚠️ Trying fallback model: {fallback_model}")
                        self.model = fallback_model
                        return self.generate(prompt, max_length)
                return None
            
            if response.status_code == 200:
                result = response.json()
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        # Try different keys that might contain the generated text
                        generated_text = result[0].get("generated_text") or result[0].get("text") or result[0].get("summary")
                        if generated_text:
                            return generated_text
                        return str(result[0])
                    return str(result[0])
                elif isinstance(result, dict):
                    # Direct dict response
                    generated_text = result.get("generated_text") or result.get("text") or result.get("summary")
                    if generated_text:
                        return generated_text
                    return str(result)
                return str(result)
            elif response.status_code == 503:
                # Model loading, wait and retry
                import time
                print("⏳ Model is loading, waiting 15 seconds...")
                time.sleep(15)
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict):
                            return result[0].get("generated_text") or result[0].get("text") or str(result[0])
                        return str(result[0])
                    elif isinstance(result, dict):
                        return result.get("generated_text") or result.get("text") or str(result)
                    return str(result)
            else:
                error_text = response.text[:500] if response.text else "Unknown error"
                print(f"⚠️ Hugging Face LLM API error ({response.status_code}): {error_text}")
            
            return None
            
        except Exception as e:
            print(f"Error calling Hugging Face LLM: {e}")
            return None


if LANGCHAIN_AVAILABLE:
    class HuggingFaceLangChainLLM(BaseLLM):
        """LangChain-compatible wrapper for Hugging Face LLM."""
        
        api_key: Optional[str] = None
        model_name: str = "google/flan-t5-base"
        hf_llm: Optional[HuggingFaceLLM] = None
        
        def __init__(self, api_key: str = None, model: str = None, **kwargs):
            """
            Initialize Hugging Face LLM for LangChain.
            
            Args:
                api_key: Hugging Face API key
                model: Model name
            """
            super().__init__(**kwargs)
            self.api_key = api_key
            self.model_name = model or "google/flan-t5-base"
            self.hf_llm = HuggingFaceLLM(api_key=api_key, model=model)
        
        @property
        def _llm_type(self) -> str:
            """Return type of LLM."""
            return "huggingface"
        
        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            """Call the LLM with the given prompt."""
            if not self.hf_llm or not self.hf_llm.is_available():
                raise ValueError("Hugging Face API key not configured")
            
            response = self.hf_llm.generate(prompt, max_length=kwargs.get("max_length", 500))
            if response is None:
                raise ValueError("Hugging Face API returned None")
            return response
        
        def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> LLMResult:
            """Generate responses for multiple prompts."""
            from langchain_core.outputs import Generation
            generations = []
            for prompt in prompts:
                text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                generations.append([Generation(text=text)])
            return LLMResult(generations=generations)
        
        def invoke(self, input, config=None, **kwargs):
            """Invoke the LLM (LangChain interface)."""
            if isinstance(input, list):
                # Handle list of messages - convert to prompt string
                prompt_parts = []
                for msg in input:
                    if hasattr(msg, "content"):
                        role = getattr(msg, "type", "human")
                        if role == "human" or role == "user":
                            prompt_parts.append(f"Human: {msg.content}")
                        elif role == "assistant" or role == "ai":
                            prompt_parts.append(f"Assistant: {msg.content}")
                        elif role == "system":
                            prompt_parts.append(f"System: {msg.content}")
                        else:
                            prompt_parts.append(str(msg.content))
                    else:
                        prompt_parts.append(str(msg))
                prompt = "\n".join(prompt_parts) + "\nAssistant:"
            elif isinstance(input, str):
                prompt = input
            elif hasattr(input, "content"):
                prompt = input.content
            else:
                prompt = str(input)
            
            response_text = self._call(prompt, **kwargs)
            
            # Return as AIMessage for compatibility
            if isinstance(input, list):
                return AIMessage(content=response_text)
            return response_text
        
        def stream(self, input, config=None, **kwargs):
            """Stream responses (not implemented, returns full response)."""
            response = self.invoke(input, config, **kwargs)
            if hasattr(response, "content"):
                yield response
            else:
                yield AIMessage(content=str(response))
else:
    class HuggingFaceLangChainLLM:
        """LangChain-compatible wrapper for Hugging Face LLM (fallback when LangChain not available)."""
        
        def __init__(self, api_key: str = None, model: str = None, **kwargs):
            self.api_key = api_key
            self.model_name = model or "google/flan-t5-base"
            self.hf_llm = HuggingFaceLLM(api_key=api_key, model=model)
        
        def invoke(self, input, config=None, **kwargs):
            """Invoke the LLM (simple interface)."""
            if isinstance(input, list):
                prompt = " ".join([str(m.content if hasattr(m, "content") else m) for m in input])
            elif isinstance(input, str):
                prompt = input
            else:
                prompt = str(input)
            
            response = self.hf_llm.generate(prompt, max_length=kwargs.get("max_length", 500))
            return response if response else ""

