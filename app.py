# app.py
import os
import torch
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import runpod
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RerankResult:
    """Result of reranking operation"""
    index: int
    score: float
    text: str

class RerankRequest(BaseModel):
    """Request model for reranking"""
    query: str
    documents: List[str]
    top_k: Optional[int] = None
    return_documents: bool = True

class RerankResponse(BaseModel):
    """Response model for reranking"""
    results: List[Dict[str, Any]]
    total_documents: int
    query: str

class BGEReranker:
    """BGE Reranker model wrapper"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self.max_length = 1024
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the tokenizer and model"""
        try:
            logger.info(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading model from {self.model_name} on {self.device}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def compute_score(self, query: str, document: str) -> float:
        """Compute relevance score between query and document"""
        try:
            # Tokenize the query-document pair
            inputs = self.tokenizer(
                query, 
                document, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get the score (logit) - higher means more relevant
                score = outputs.logits.squeeze().float().cpu().item()
                
            return score
            
        except Exception as e:
            logger.error(f"Error computing score: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[RerankResult]:
        """Rerank documents based on relevance to query"""
        if not documents:
            return []
        
        logger.info(f"Reranking {len(documents)} documents for query: {query[:100]}...")
        
        results = []
        
        # Compute scores for all documents
        for idx, doc in enumerate(documents):
            try:
                score = self.compute_score(query, doc)
                results.append(RerankResult(index=idx, score=score, text=doc))
            except Exception as e:
                logger.warning(f"Error processing document {idx}: {e}")
                # Add with very low score if processing fails
                results.append(RerankResult(index=idx, score=float('-inf'), text=doc))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply top_k if specified
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        
        logger.info(f"Reranking completed. Top score: {results[0].score if results else 'N/A'}")
        return results

# Global reranker instance
reranker = None

def initialize_model():
    """Initialize the reranker model"""
    global reranker
    try:
        logger.info("Initializing BGE Reranker...")
        reranker = BGEReranker()
        reranker.load_model()
        logger.info("Model initialization completed")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False

# FastAPI app for HTTP endpoint
app = FastAPI(
    title="BGE Reranker Service",
    description="Reranking service using BAAI/bge-reranker-v2-m3",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    if not initialize_model():
        raise RuntimeError("Failed to initialize reranker model")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": reranker.model_name if reranker else "not loaded",
        "device": reranker.device if reranker else "unknown"
    }

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Rerank documents based on query relevance"""
    if not reranker:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    if len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Perform reranking
        results = reranker.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k
        )
        
        # Format response
        formatted_results = []
        for result in results:
            item = {
                "index": result.index,
                "relevance_score": result.score
            }
            if request.return_documents:
                item["text"] = result.text
            formatted_results.append(item)
        
        return RerankResponse(
            results=formatted_results,
            total_documents=len(request.documents),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")

# RunPod handler function
def runpod_handler(job):
    """RunPod serverless handler"""
    global reranker
    
    if not reranker:
        if not initialize_model():
            return {"error": "Failed to initialize model"}
    
    try:
        job_input = job.get("input", {})
        
        query = job_input.get("query", "")
        documents = job_input.get("documents", [])
        top_k = job_input.get("top_k")
        return_documents = job_input.get("return_documents", True)
        
        if not query or not documents:
            return {"error": "Query and documents are required"}
        
        # Perform reranking
        results = reranker.rerank(query=query, documents=documents, top_k=top_k)
        
        # Format response
        formatted_results = []
        for result in results:
            item = {
                "index": result.index,
                "relevance_score": result.score
            }
            if return_documents:
                item["text"] = result.text
            formatted_results.append(item)
        
        return {
            "results": formatted_results,
            "total_documents": len(documents),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Error in RunPod handler: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Check if running in RunPod serverless mode
    if os.getenv("RUNPOD_ENDPOINT_ID"):
        logger.info("Starting RunPod serverless handler")
        runpod.serverless.start({"handler": runpod_handler})
    else:
        # Run as HTTP server
        logger.info("Starting HTTP server")
        initialize_model()
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=int(os.getenv("PORT", 8000)),
            log_level="info"
        )