import logging
from typing import Dict, Any

from anime_rec_engine.llm_models import EmbeddingModel, GroqModel
from anime_rec_engine.vector_store import VectorStore
from anime_rec_engine.prompts import PROMPT_TEMPLATE

class Recommender:
    """
    Orchestrates the entire recommendation process.
    """
    def get_recommendation(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Generates an anime recommendation based on a user query.
        
        Args:
            query (str): The user's query describing what they want to watch.
            n_results (int): The number of similar animes to fetch for context.
            
        Returns:
            Dict[str, Any]: A dictionary containing the LLM's response and the source animes.
        """
        logging.info(f"Generating embedding for query: '{query}'")
        query_embedding = EmbeddingModel.get_embeddings(texts=[query])
        
        if not query_embedding:
            logging.error("Failed to generate query embedding.")
            return {"llm_response": "Sorry, I couldn't process your request at the moment.", "source_animes": []}
            
        logging.info(f"Querying vector store for {n_results} similar animes.")
        similar_animes = VectorStore.query_animes(
            query_embedding=query_embedding[0],
            n_results=n_results
        )
        
        if not similar_animes:
            logging.warning("No similar animes found in the vector store.")
            return {"llm_response": "I couldn't find any anime matching your description in my database.", "source_animes": []}

        logging.info("Creating prompt for the LLM.")
        prompt = PROMPT_TEMPLATE.create_prompt(query=query, context=similar_animes)
        
        logging.info("Requesting recommendation from Groq LLM.")
        llm_response = GroqModel.get_response(prompt=prompt)

        return {"llm_response": llm_response, "source_animes": similar_animes}

# Create a singleton instance
RECOMMENDER = Recommender()

