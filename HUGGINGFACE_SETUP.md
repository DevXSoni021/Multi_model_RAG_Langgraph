# Hugging Face Fallback Setup

## Overview

The system now includes a Hugging Face fallback for multimodal image understanding. This provides an alternative to OpenAI when:
- OpenAI rate limits are hit
- You want to use open-source models
- Cost optimization is needed

## Configuration

### 1. Get Hugging Face API Key

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Copy the token

### 2. Add to .env file

```env
HUGGINGFACE_API_KEY=hf_your_token_here
```

## Available Models

The system uses `Salesforce/blip-image-captioning-large` by default. You can change this in `config.py`:

```python
HUGGINGFACE_MULTIMODAL_MODEL = "Salesforce/blip-image-captioning-large"
```

### Recommended Models:

1. **Salesforce/blip-image-captioning-large** (Default)
   - Best quality descriptions
   - Good for detailed image understanding
   - ~1-2 seconds per image

2. **Salesforce/blip-image-captioning-base**
   - Faster processing
   - Good quality
   - ~0.5-1 second per image

3. **nlpconnect/vit-gpt2-image-captioning**
   - Balanced speed/quality
   - Good for general use

4. **microsoft/git-base**
   - Detailed descriptions
   - Good for complex images

## How It Works

1. **Image Processing**: When images are retrieved from PDFs, they can be described using Hugging Face models
2. **Automatic Fallback**: If OpenAI rate limits are hit, the system can fall back to Hugging Face for text generation
3. **Image Descriptions**: Images are automatically described when retrieved, providing context for the LLM

## Usage

The Hugging Face fallback is automatically used when:
- Images are retrieved and need descriptions
- OpenAI rate limits are encountered
- You explicitly enable it in config

## Benefits

- **No Rate Limits**: Hugging Face free tier has generous limits
- **Open Source**: Uses open-source models
- **Cost Effective**: Free tier available
- **Multimodal**: Can understand both images and text

## Limitations

- Slower than OpenAI GPT-4 Vision
- May have lower quality for complex images
- Some models require warm-up time on first use

## Testing

To test the Hugging Face integration:

```python
from huggingface_fallback import HuggingFaceMultimodal

hf = HuggingFaceMultimodal()
if hf.is_available():
    description = hf.describe_image(image_base64, "What is in this image?")
    print(description)
```

