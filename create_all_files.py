#!/usr/bin/env python3
"""Helper script to create all Python files for Colab notebook."""
import sys

# Read source files and create write statements
files = {
    'pdf_processor.py': 'pdf_processor.py',
    'vector_store.py': 'vector_store.py',
    'huggingface_fallback.py': 'huggingface_fallback.py',
    'agents.py': 'agents.py',
    'image_embeddings.py': 'image_embeddings.py'
}

for target, source in files.items():
    try:
        with open(source, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Escape for Python string
        content_escaped = content.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
        
        # Create write statement
        write_code = f'''with open("{target}", "w", encoding="utf-8") as f:
    f.write("""{content_escaped}""")
print(f"✅ Created {target}")'''
        
        print(f"# Cell for {target}:")
        print(write_code)
        print()
    except Exception as e:
        print(f"Error reading {source}: {e}")

