try:
    from langchain_classic.chains import RetrievalQA
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_ibm import WatsonxLLM
    print("✅ Success! All libraries are installed and resolving.")
except ImportError as e:
    print(f"Error: {e}")