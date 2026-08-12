"""
Hugging Face Spaces Entrypoint
Runs the main MULTIMEDIA AUTHENTICITY LAB application (deepfake_detectorV3.py)
"""
import runpy
import sys
from pathlib import Path

# Ensure working directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Execute the main detector script
runpy.run_path("deepfake_detectorV3.py", run_name="__main__")
