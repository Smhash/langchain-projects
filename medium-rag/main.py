import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Configuration constants
DEFAULT_QUERY = "What are the top 3 lessons learned from Building AI Agents in LegalTech?"
RETRIEVAL_PROMPT_HUB_ID = "langchain-ai/retrieval-qa-chat"

load_dotenv()


def setup_embeddings() -> OpenAIEmbeddings:
    """Initialize and return OpenAI embeddings."""
    return OpenAIEmbeddings()


def setup_llm() -> ChatOpenAI:
    """Initialize and return ChatOpenAI model."""
    return ChatOpenAI()


def setup_vectorstore(embeddings: OpenAIEmbeddings) -> PineconeVectorStore:
    """Initialize and return Pinecone vector store."""
    index_name = os.environ.get("INDEX_NAME")
    if not index_name:
        raise ValueError("INDEX_NAME environment variable is required")
    
    return PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings
    )


def create_retrieval_chain_with_prompt(llm: ChatOpenAI) -> Any:
    """Create retrieval chain with QA chat prompt."""
    retrieval_qa_chat_prompt = hub.pull(RETRIEVAL_PROMPT_HUB_ID)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    return combine_docs_chain


def query_vectorstore(vectorstore: PineconeVectorStore, combine_docs_chain: Any, query: str) -> Dict[str, Any]:
    """Query the vector store and return results."""
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), 
        combine_docs_chain=combine_docs_chain
    )
    return retrieval_chain.invoke(input={"input": query})


def main():
    """Main function to run the RAG retrieval system."""
    print("Retrieving...")
    
    try:
        # Initialize components
        embeddings = setup_embeddings()
        llm = setup_llm()
        vectorstore = setup_vectorstore(embeddings)
        
        # Create retrieval chain
        combine_docs_chain = create_retrieval_chain_with_prompt(llm)
        
        # Query the system
        query = DEFAULT_QUERY
        result = query_vectorstore(vectorstore, combine_docs_chain, query)
        
        print("Query:", query)
        print("Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
