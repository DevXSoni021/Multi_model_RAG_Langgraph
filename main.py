"""Main entry point for the application."""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # Run Streamlit app
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())

