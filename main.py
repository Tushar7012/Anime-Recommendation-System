from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

from anime_rec_engine.recommender import RECOMMENDER

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="AI Anime Recommendation API",
    description="An API that provides anime recommendations based on user queries using a content-based filtering approach with AI embeddings.",
    version="1.0.0"
)

class RecommendationQuery(BaseModel):
    """Pydantic model for the recommendation request body."""
    query: str = Field(..., min_length=3, description="The user's query describing the kind of anime they want to watch.")
    n_results: int = Field(10, gt=0, le=20, description="The number of similar animes to retrieve for generating the recommendation.")

@app.post("/recommend/", tags=["Recommendations"])
def get_anime_recommendation(request: RecommendationQuery):
    """
    Accepts a user query and returns an AI-generated anime recommendation.
    """
    try:
        logging.info(f"Received recommendation request for query: '{request.query}'")
        recommendation = RECOMMENDER.get_recommendation(
            query=request.query,
            n_results=request.n_results
        )
        if not recommendation or not recommendation.get("llm_response"):
             raise HTTPException(status_code=404, detail="Could not find a suitable recommendation based on your query.")
        
        return recommendation
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the AI Anime Recommendation API!"}

