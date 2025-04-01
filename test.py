import sys
print(f"Using Python at: {sys.executable}")

try:
    from langchain_community.vectorstores import FAISS
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")