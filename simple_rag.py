from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from data_ingestion import initialize_embeddings, initialize_vector_store, data_ingestion

import torch
import os

class SimpleRAG:
    """Simple RAG system using LangChain and HuggingFace transformers."""
    
    def __init__(self, 
                 model_id: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_new_tokens: int = 200,
                 retrieval_k: int = 3):
        """
        Initialize the SimpleRAG system.
        
        Args:
            model_id: HuggingFace model identifier
            max_new_tokens: Maximum new tokens to generate
            retrieval_k: Number of documents to retrieve
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.retrieval_k = retrieval_k
        
        # Check GPU availability
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        
        # Set default template if none provided
        self.template = """
        You are a helpful assistant that answers questions using only the provided context.

        CRITICAL INSTRUCTIONS:
        - Answer ONLY using information from the context below
        - If the answer is not in the context, respond with "I don't know"
        - You MUST respond with EXACTLY this JSON format - no additional text before or after:

        {{"answer": "your complete answer here"}}

        Context: {context}

        Question: {question}

        Response:
        """
        
        # Initialize components
        self._initialize_model()
        self._setup_vector_store()
        self._setup_rag_chain()
    
    def _initialize_model(self):
        """Initialize the LLM model with quantization config."""
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True
        )
        
        # Create generation pipeline
        gen_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            do_sample=False,
            max_new_tokens=self.max_new_tokens
        )
        
        # Create LangChain-compatible LLM
        self.llm = HuggingFacePipeline(pipeline=gen_pipe)
    
    def _setup_vector_store(self):
        """Initialize embeddings and vector store, run data ingestion if needed."""
        
        # Initialize embeddings and vector store
        self.embeddings = initialize_embeddings()
        self.vectorstore = initialize_vector_store(self.embeddings)
        
        # Check if vector store needs data ingestion
        persist_dir = "./chroma_db"
        if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
            print("No existing vector store found. Running data ingestion...")
            results = data_ingestion()
            if results["status"] == "success":
                print(f"Data ingestion completed: {results['chunks_count']} chunks processed")
            else:
                print(f"Data ingestion failed: {results['message']}")
                raise RuntimeError(f"Data ingestion failed: {results['message']}")
        else:
            print("Using existing ChromaDB vectorstore")
        
        # Set up retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.retrieval_k})
    
    def _setup_rag_chain(self):
        """Set up the RAG chain with retriever, prompt, LLM, and output parser."""
        
        # Create prompt template
        self.prompt = PromptTemplate.from_template(self.template)
        
        # Create RAG chain
        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    @staticmethod
    def _format_docs(docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join(d.page_content for d in docs)
    
    def query(self, question: str) -> dict:
        """
        Process a query through the RAG system and return both answer and context.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing:
            - answer: The RAG system's answer as a string
            - contexts: List of retrieved document texts
            - source_documents: List of retrieved document objects
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.invoke(question)
        
        # Generate answer using the RAG chain
        answer = self.rag_chain.invoke(question)
        
        return {
            "answer": answer.strip(),
            "contexts": [doc.page_content for doc in retrieved_docs],
            "source_documents": retrieved_docs
        }
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model and configuration."""
        return {
            "model_id": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "retrieval_k": self.retrieval_k,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "model_id": self.model_id
        }

def main():
    """Main function demonstrating basic usage of the SimpleRAG class."""
    # Initialize RAG system
    rag = SimpleRAG()
    
    # Example query
    user_q = "How do I create a matrix filled with 0s?"
    result = rag.query(user_q)
    print(f"\n=== RAG Answer ===\n{result['answer']}")
    print(f"\n=== Retrieved Contexts ===\n{len(result['contexts'])} contexts retrieved")
    
    # Print model info
    print("\n=== Model Info ===\n", rag.get_model_info())


if __name__ == "__main__":
    main()
