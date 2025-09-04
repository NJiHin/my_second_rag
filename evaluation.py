from simple_rag import SimpleRAG
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic,Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from data_ingestion import initialize_embeddings

import torch
import asyncio
import pandas as pd
import time

async def evaluate():
    # Start timer
    start_time = time.time()

    # Check GPU availability
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    
    # Initialize RAG system first
    rag_system = SimpleRAG()
    
    # Reuse the model and tokenizer from SimpleRAG to save memory
    model_info = rag_system.get_model_info()
    
    # Create evaluation pipeline using the same model instance
    # Modified parameters for better RAGAS compatibility
    gen = pipeline(
        "text-generation",
        model=model_info["model"],
        tokenizer=model_info["tokenizer"],
        return_full_text=False,
        do_sample=True,
        temperature=0.1,
        max_new_tokens=300,
        pad_token_id=model_info["tokenizer"].eos_token_id) #what

    llm = HuggingFacePipeline(pipeline=gen)
    evaluator_llm = LangchainLLMWrapper(llm)
    
    # Initialize embeddings using same model as RAG system
    embeddings = initialize_embeddings()
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    # Load QA dataset from CSV (ANSI encoded)
    qa_df = pd.read_csv("QA.csv", encoding='cp1252')
    print(f"Loaded {len(qa_df)} QA pairs from CSV")
    
    # Create RAGAS dataset samples
    samples = []
    results = []
    
    # Initialize RAGAS metrics with evaluator LLM and embeddings outside the loop
    #answer_correctness_metric = answer_correctness.init(llm=evaluator_llm)
    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    answer_relevancy_metric = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    context_precision_metric = LLMContextPrecisionWithReference(llm=evaluator_llm)
    '''
    # Define custom metrics for MATLAB-specific evaluation (once, outside the loop)
    syntax_critic = AspectCritic(
        name="matlab_syntax",
        llm=evaluator_llm,
        definition="""Evaluate MATLAB code syntax correctness. Check for:
        - Proper function declarations and end statements
        - Correct variable naming conventions
        - Valid MATLAB operators and built-in functions
        - Proper array indexing (1-based)
        - Semicolon usage for output suppression
        Score: 1 if syntactically valid, 0 if contains syntax errors."""
    )
    '''
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

    for _, row in qa_df.iterrows():
        question = row['Question']
        reference_answer = row['Answer']
        
        # Get response and context from your RAG system
        rag_result = rag_system.query(question)
        
        # Create RAGAS SingleTurnSample with reference answer
        sample = SingleTurnSample(
            user_input=question,
            response=rag_result["answer"],
            retrieved_contexts=rag_result["contexts"],
            reference=reference_answer
        )
        samples.append(sample)
    
        
        # Evaluate with all metrics
        try:
            faithfulness_score = await faithfulness_metric.single_turn_ascore(sample)
            answer_relevancy_score = await answer_relevancy_metric.single_turn_ascore(sample)
            context_precision_score = await context_precision_metric.single_turn_ascore(sample)
            #matlab_syntax_score = await syntax_critic.single_turn_ascore(sample)
            matlab_logic_score = await semantic_critic.single_turn_ascore(sample)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            faithfulness_score = None
            answer_relevancy_score = None
            context_precision_score = None
            #matlab_syntax_score = None
            matlab_logic_score = None
        
        result = {
            "question": question,
            "generated_answer": rag_result["answer"],
            "reference_answer": reference_answer,
            "retrieved_contexts": "; ".join(rag_result["contexts"]),  # Join contexts for CSV compatibility
            "faithfulness": faithfulness_score,
            "answer_relevancy": answer_relevancy_score,
            "context_precision": context_precision_score,
            #"matlab_syntax": matlab_syntax_score,
            "matlab_logic": matlab_logic_score
        }
        results.append(result)
        
        # Print evaluation metrics after each question
        print(f"Processed question {len(results)}/{len(qa_df)}: {question[:60]}...")
        print(f"  Faithfulness: {faithfulness_score if faithfulness_score is not None else 'Failed'}")
        print(f"  Answer Relevancy: {answer_relevancy_score if answer_relevancy_score is not None else 'Failed'}")
        print(f"  Context Precision: {context_precision_score if context_precision_score is not None else 'Failed'}")
        #print(f"  MATLAB Syntax: {matlab_syntax_score if matlab_syntax_score is not None else 'Failed'}")
        print(f"  MATLAB Logic: {matlab_logic_score if matlab_logic_score is not None else 'Failed'}")
        print("-" * 70)
    
    # Convert results to pandas DataFrame and save
    df_results = pd.DataFrame(results)
    df_results.to_csv("evaluation_results.csv", index=False, encoding='utf-8')
    
    # Calculate and display total time
    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Total time: {int(minutes)}m {seconds:.1f}s")

    return df_results   

df_results = asyncio.run(evaluate())