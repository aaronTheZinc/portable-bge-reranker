import os
import torch
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
                device_map="auto" if self.device == "cuda" else None,
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
        for idx, doc in enumerate(documents):
            try:
                score = self.compute_score(query, doc)
                results.append(RerankResult(index=idx, score=score, text=doc))
            except Exception as e:
                logger.warning(f"Error processing document {idx}: {e}")
                results.append(RerankResult(index=idx, score=float("-inf"), text=doc))

        results.sort(key=lambda x: x.score, reverse=True)

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


# FastAPI app
app = FastAPI(
    title="BGE Reranker Service",
    description="Reranking service using BAAI/bge-reranker-v2-m3",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    if not initialize_model():
        raise RuntimeError("Failed to initialize reranker model")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": reranker.model_name if reranker else "not loaded",
        "device": reranker.device if reranker else "unknown",
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    if not reranker:
        raise HTTPException(status_code=500, detail="Model not initialized")

    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    if len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = reranker.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k,
        )

        formatted_results = []
        for result in results:
            item = {
                "index": result.index,
                "relevance_score": result.score,
            }
            if request.return_documents:
                item["text"] = result.text
            formatted_results.append(item)

        return RerankResponse(
            results=formatted_results,
            total_documents=len(request.documents),
            query=request.query,
        )

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")
