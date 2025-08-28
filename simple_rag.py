from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from data_ingestion import initialize_embeddings, initialize_vector_store, data_ingestion

import torch
import os

print("GPU:", torch.cuda.get_device_name(0))

# define and set up the model
model_id = "Qwen/Qwen2.5-7B-Instruct"

# bitsandbytes 8-bit config
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True) # use same tokenizer as per best practice

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only = True
)

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    return_full_text=False,
    do_sample=False, #
    max_new_tokens=200
)

# so that we can use llm.invoke() (langchain style)
llm = HuggingFacePipeline(pipeline=gen_pipe)

# initialize embeddings and vector store using data_ingestion module
embeddings = initialize_embeddings()
vectorstore = initialize_vector_store(embeddings)

# check if vector store is empty and needs data ingestion
persist_dir = "./chroma_db"
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    print("No existing vector store found. Running data ingestion...")
    results = data_ingestion()
    if results["status"] == "success":
        print(f"Data ingestion completed: {results['chunks_count']} chunks processed")
    else:
        print(f"Data ingestion failed: {results['message']}")
else:
    print("Using existing ChromaDB vectorstore")

# set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# define rag template
template = """You are a helpful assistant. Answer the question using ONLY the context.
If the answer isn't in the context, say you don't know. Return ONLY MATLAB CODE.

# Context
{context}

# Question
{question}

# Answer:"""
prompt = PromptTemplate.from_template(template)

# why
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_q = "How do I create a matrix filled with 0s?"
answer = rag_chain.invoke(user_q)
print("\n=== RAG Answer ===\n", answer)
