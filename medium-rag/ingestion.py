import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone

# Configuration constants
SOURCE_FILE = "medium-building-ai-agents-in-legaltech.txt"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0

load_dotenv()

def load_document(file_path: str) -> List:
    """Load document from file."""
    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path)
    return loader.load()


def split_document(document: List) -> List:
    """Split document into chunks."""
    print("Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(document)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_index_if_not_exists() -> None:
    """Create Pinecone index if it doesn't exist."""
    print("Checking Pinecone index...")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    index_name = os.environ.get("INDEX_NAME")
    if not index_name:
        raise ValueError("INDEX_NAME environment variable is required")
    
    if index_name not in pc.list_indexes().names():
        print(f"Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embeddings dimension
            metric="cosine"
        )
        print(f"Index '{index_name}' created successfully!")
    else:
        print(f"Index '{index_name}' already exists")


def setup_embeddings() -> OpenAIEmbeddings:
    """Initialize OpenAI embeddings."""
    print("Setting up embeddings...")
    return OpenAIEmbeddings() # Automatically reads OPENAI_API_KEY from env


def ingest_to_pinecone(chunks: List, embeddings: OpenAIEmbeddings) -> None:
    """Ingest document chunks to Pinecone vector store."""
    index_name = os.environ.get("INDEX_NAME")
    if not index_name:
        raise ValueError("INDEX_NAME environment variable is required")
    
    print(f"Ingesting {len(chunks)} chunks to Pinecone index '{index_name}'...")
    PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=index_name
    )
    print("Ingestion completed successfully!")


def main():
    """Main function to run the document ingestion process."""
    try:
        # Create index if it doesn't exist
        create_index_if_not_exists()
        
        # Load document
        document = load_document(SOURCE_FILE)
        
        # Split into chunks
        chunks = split_document(document)
        
        # Setup embeddings
        embeddings = setup_embeddings()
        
        # Ingest to Pinecone
        ingest_to_pinecone(chunks, embeddings)
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
