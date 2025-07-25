#!/usr/bin/env python3
"""
Quick launcher for Job Trend Analyzer Web UI
Alternative to run_ui.bat for Python environments
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit web UI"""
    print("\n" + "="*50)
    print("   Job Trend Analyzer Web UI")
    print("="*50)
    
    # Check if streamlit is available
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit is not installed")
        print("Please install requirements: pip install -r requirements.txt")
        return 1
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ui_file = os.path.join(script_dir, "src", "web_ui.py")
    
    if not os.path.exists(ui_file):
        print(f"❌ UI file not found: {ui_file}")
        return 1
    
    print("\n🚀 Starting Web UI...")
    print("📍 Open your browser and go to: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", ui_file,
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Web UI...")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
