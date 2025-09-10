#!/usr/bin/env python3
"""
Main entry point for running the Wafer Clustering Agent UI
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

def main():
    """Launch the Gradio UI"""
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("Please set it using: export OPENAI_API_KEY='your-key-here'")
        print("Or create a .env file with: OPENAI_API_KEY=your-key-here")
        print()
    
    # Import and launch UI
    from src.ui import create_gradio_interface
    
    print("🚀 Starting Semiconductor Wafer Clustering Agent UI...")
    print("📊 Open your browser to the URL shown below")
    print()
    
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(
        share=True,  # Create public link
        server_name="0.0.0.0",  # Accept connections from any IP
        server_port=7860,  # Default Gradio port
        debug=False,  # Set to True for debugging
        show_error=True
    )

if __name__ == "__main__":
    main()