import os

# PATHS
# Define the root directory of your project.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(ROOT_DIR, ".cache", "docling", "models") # Path to store Docling artifacts.

# DOCUMENT PROCESSING PARAMETERS
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200