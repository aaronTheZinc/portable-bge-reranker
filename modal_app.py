import os
import torch
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import modal

# ------------------------------
# Modal setup
# ------------------------------
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "fastapi[standard]",
    "pydantic",
    "accelerate",  # Required for device_map="auto"
)

app = modal.App("bge-reranker", image=image)

# ------------------------------
# Pydantic Models
# ------------------------------
@dataclass
class RerankResult:
    index: int
    score: float
    text: str


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None
    return_documents: bool = True

    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v):
        if not v:
            raise ValueError('Documents list cannot be empty')
        if len(v) > 1000:  # Reasonable limit
            raise ValueError('Too many documents (max 1000)')
        return v

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValueError('top_k must be positive')
        return v


class RerankResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_documents: int
    query: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    environment: str


# ------------------------------
# BGE Reranker Class
# ------------------------------
class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self.max_length = 1024
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        self.logger.info(f"Loading tokenizer from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.logger.info(f"Loading model from {self.model_name} on {self.device}")
        
        # Load model with proper device handling
        if self.device == "cuda":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",  # This requires accelerate
            )
        else:
            # For CPU, load normally without device_map
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )
            self.model = self.model.to(self.device)
            
        self.model.eval()
        self.logger.info("Model loaded successfully")

    def compute_score(self, query: str, document: str) -> float:
        inputs = self.tokenizer(
            query,
            document,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.logits.squeeze().float().cpu().item()

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[RerankResult]:
        results = []
        for idx, doc in enumerate(documents):
            try:
                score = self.compute_score(query, doc)
                results.append(RerankResult(idx, score, doc))
            except Exception as e:
                self.logger.warning(f"Error processing document {idx}: {e}")
                results.append(RerankResult(idx, float("-inf"), doc))
        results.sort(key=lambda x: x.score, reverse=True)
        if top_k and top_k > 0:
            results = results[:top_k]
        return results


# Global reranker instance
reranker = None


def get_reranker():
    """Initialize reranker if not already loaded"""
    global reranker
    if reranker is None:
        reranker = BGEReranker()
        reranker.load_model()
    return reranker


# ------------------------------
# Modal FastAPI Endpoints
# ------------------------------

@app.function(
    gpu="A10G",
    memory=8192,
    timeout=3600,
)
@modal.fastapi_endpoint(method="GET")
def root():
    """Root endpoint"""
    return JSONResponse({
        "message": "BGE Reranker Service",
        "status": "online",
        "environment": "modal"
    })


@app.function(
    gpu="A10G", 
    memory=8192,
    timeout=3600,
)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint"""
    try:
        reranker_instance = get_reranker()
        return JSONResponse({
            "status": "healthy",
            "model": reranker_instance.model_name,
            "device": reranker_instance.device,
            "environment": "modal"
        })
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "environment": "modal"
            }
        )


@app.function(
    gpu="A10G",
    memory=8192, 
    timeout=3600,
)
@modal.fastapi_endpoint(method="POST")
def rerank(request: RerankRequest):
    """Main reranking endpoint"""
    try:
        reranker_instance = get_reranker()
        results = reranker_instance.rerank(request.query, request.documents, request.top_k)
        
        formatted_results = [
            {"index": r.index, "relevance_score": r.score, "text": r.text}
            for r in results
        ]
        
        response = RerankResponse(
            results=formatted_results,
            total_documents=len(request.documents),
            query=request.query
        )
        
        return JSONResponse(response.dict())
        
    except Exception as e:
        logging.error(f"Error during reranking: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error during reranking", "detail": str(e)}
        )


# ------------------------------
# Local testing support
# ------------------------------
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="BGE Reranker Service - Local Testing")
    parser.add_argument("--test", action="store_true", help="Run a quick test")
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing BGE Reranker locally...")
        try:
            # Initialize reranker
            local_reranker = BGEReranker()
            local_reranker.load_model()
            
            # Test reranking
            query = "machine learning algorithms"
            docs = [
                "Deep learning is a subset of machine learning",
                "The weather is nice today",
                "Neural networks are powerful ML models",
                "I like pizza for dinner",
                "Random forests are ensemble methods in ML"
            ]
            
            results = local_reranker.rerank(query, docs, top_k=3)
            print("✅ Test successful!")
            print(f"Query: {query}")
            print("Top results:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. Score: {r.score:.4f} - {r.text}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            sys.exit(1)
    else:
        print("For local testing, run: python modal_app.py --test")
        print("To deploy to Modal, run: modal deploy modal_app.py")
        print("After deployment, your endpoints will be available at:")
        print("  - GET  /root")
        print("  - GET  /health")  
        print("  - POST /rerank")