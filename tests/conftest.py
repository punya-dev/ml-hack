import os
import sys

# Ensure code/business_entity_resolution is in sys.path for test imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT_DIR, "code", "business_entity_resolution")
if os.path.isdir(CODE_DIR) and CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
