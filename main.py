from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from simple_rag import SimpleRAG

app = FastAPI()

llm = SimpleRAG()

# Request and Response Models
class QuestionRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str
    contexts: list[str] = None

@app.get("/")
def root():
    return {"Hello":"world"}

@app.post("/generate", response_model=RAGResponse)
def generate(request: QuestionRequest):
    try:
        result = llm.query(request.question)
        return RAGResponse(
            answer=result["answer"],
            contexts=result["contexts"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG system error: {str(e)}")