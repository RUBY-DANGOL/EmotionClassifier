#!/usr/bin/env python3
"""
Simple runner script for the Flask image classifier application.
"""

import os
import sys

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set environment variables with absolute paths if needed
    os.environ.setdefault("MODEL_PATH", os.path.join(script_dir, "densenetmodel_classify.h5"))
    os.environ.setdefault("LABELS_PATH", os.path.join(script_dir, "labels.json"))
    os.environ.setdefault("TOP_K", "5")
    
    # Change to the script directory to ensure relative paths work
    os.chdir(script_dir)
    
   
    from app import app
    
    print("=" * 50)
    print("🚀 Starting Image Classifier Flask App")
    print("=" * 50)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🤖 Model file: {os.environ.get('MODEL_PATH')}")
    print(f"🏷️  Labels file: {os.environ.get('LABELS_PATH')}")
    print(f"🔢 Top predictions: {os.environ.get('TOP_K')}")
    print("=" * 50)
    print("📱 Open your browser and go to: http://localhost:5000")
    print("=" * 50)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
