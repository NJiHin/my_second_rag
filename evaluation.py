from simple_rag import SimpleRAG
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic,Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from data_ingestion import initialize_embeddings

import torch
import gc
import asyncio
import pandas as pd
import time
import os

def get_answers():
    """Generate answers using SimpleRAG and return structured data for evaluation."""
    start_time = time.time()
    
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Initialize RAG system
    print("Loading RAG system (Qwen2.5-7B)...")
    rag_system = SimpleRAG()

    # Load QA dataset from CSV (ANSI encoded)
    qa_df = pd.read_csv("QA.csv", encoding='cp1252')
    print(f"Loaded {len(qa_df)} QA pairs from CSV")

    answers_data = []

    for i, row in qa_df.iterrows():
        question = row['Question']
        reference_answer = row['Answer']
        
        print(f"Processing question {i+1}/{len(qa_df)}: {question[:60]}...")
        
        # Get response and context from RAG system
        rag_result = rag_system.query(question)
        
        # Store all data needed for evaluation
        answer_data = {
            "question": question,
            "generated_answer": rag_result["answer"],
            "reference_answer": reference_answer,
            "retrieved_contexts": rag_result["contexts"],
            "source_documents": rag_result["source_documents"]
        }
        answers_data.append(answer_data)
        
        print(f"  Generated answer: {rag_result['answer'][:100]}...")
    
    print(f"Answer generation completed in {time.time() - start_time:.1f}s")
    
    # Clean up RAG system and free GPU memory
    print("Shutting down RAG system and clearing GPU memory...")
    del rag_system
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"GPU memory cleared. Generated {len(answers_data)} answer sets.")
    return answers_data


async def evaluate(answers_data):
    """Evaluate generated answers using a separate evaluation model."""
    start_time = time.time()

    # Check GPU availability
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Load evaluation model (Mistral-7B)
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True
    )
    
    # Create evaluation pipeline using the loaded model instance
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        do_sample=True,
        temperature=0.1,
        max_new_tokens=300,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=gen)
    evaluator_llm = LangchainLLMWrapper(llm)
    
    # Initialize embeddings using same model as RAG system
    embeddings = initialize_embeddings()
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    # Initialize RAGAS metrics with evaluator LLM and embeddings
    print("Initializing evaluation metrics...")
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    answer_relevancy_metric = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    context_precision_metric = LLMContextPrecisionWithReference(llm=evaluator_llm)
    
    semantic_critic = AspectCritic(
        name="matlab_logic",
        llm=evaluator_llm, 
        definition="""Evaluate logical correctness of MATLAB code:
        - Does the algorithm solve the stated problem?
        - Are mathematical operations correct?
        - Is the approach computationally efficient for MATLAB?
        - Are edge cases handled appropriately?
        Score: 1-5 scale based on logical soundness."""
    )
    
    # Process each answer for evaluation
    results = []
    print(f"Evaluating {len(answers_data)} generated answers...")
    
    for i, answer_data in enumerate(answers_data):
        print(f"Evaluating answer {i+1}/{len(answers_data)}: {answer_data['question'][:60]}...")
        
        # Create RAGAS SingleTurnSample
        sample = SingleTurnSample(
            user_input=answer_data["question"],
            response=answer_data["generated_answer"],
            retrieved_contexts=answer_data["retrieved_contexts"],
            reference=answer_data["reference_answer"]
        )
        
        # Evaluate with all metrics
        try:
            faithfulness_score = await faithfulness_metric.single_turn_ascore(sample)
            answer_relevancy_score = await answer_relevancy_metric.single_turn_ascore(sample)
            context_precision_score = await context_precision_metric.single_turn_ascore(sample)
            matlab_logic_score = await semantic_critic.single_turn_ascore(sample)
        except Exception as e:
            print(f"Evaluation failed for question {i+1}: {e}")
            faithfulness_score = None
            answer_relevancy_score = None
            context_precision_score = None
            matlab_logic_score = None
        
        # Store results
        result = {
            "question": answer_data["question"],
            "generated_answer": answer_data["generated_answer"],
            "reference_answer": answer_data["reference_answer"],
            "retrieved_contexts": "; ".join(answer_data["retrieved_contexts"]),
            "faithfulness": faithfulness_score,
            "answer_relevancy": answer_relevancy_score,
            "context_precision": context_precision_score,
            "matlab_logic": matlab_logic_score
        }
        results.append(result)
        
        # Print evaluation metrics
        print(f"  Faithfulness: {faithfulness_score if faithfulness_score is not None else 'Failed'}")
        print(f"  Answer Relevancy: {answer_relevancy_score if answer_relevancy_score is not None else 'Failed'}")
        print(f"  Context Precision: {context_precision_score if context_precision_score is not None else 'Failed'}")
        print(f"  MATLAB Logic: {matlab_logic_score if matlab_logic_score is not None else 'Failed'}")
        print("-" * 70)
    
    # Convert results to pandas DataFrame and save
    df_results = pd.DataFrame(results)
    df_results.to_csv("evaluation_results.csv", index=False, encoding='utf-8')
    
    # Calculate and display total time
    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Evaluation completed in {int(minutes)}m {seconds:.1f}s")
    
    # Clean up evaluation model and free GPU memory
    print("Cleaning up evaluation model and clearing GPU memory...")
    del model
    del tokenizer
    del gen
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return df_results   

def main():
    """Main function to orchestrate sequential model loading evaluation."""
    print("Generating answers")
    answers_data = get_answers()
    
    print("Evaluating answers")
    df_results = asyncio.run(evaluate(answers_data))
    return df_results

if __name__ == "__main__":
    main()