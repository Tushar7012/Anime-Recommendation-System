from .llm_models import embedding_model, groq_model
from .vector_store import vector_store
from .prompts import prompt_template
from typing import List, Dict, Any

class Recommender:
    """
    The main class for generating anime recommendations.
    
    This class orchestrates the entire recommendation pipeline:
    1. Creates an embedding from the user's query.
    2. Queries the vector store to find similar animes.
    3. Creates a prompt with the retrieved context.
    4. Calls the Groq LLM to generate the final recommendation text.
    """
    
    def __init__(self):
        """
        Initializes the Recommender with instances of the required components.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.prompt_template = prompt_template
        self.groq_model = groq_model

    def get_recommendation(self, user_query: str, n_results: int = 10) -> str:
        """
        Generates an anime recommendation based on a user's query.

        Args:
            user_query (str): The user's request (e.g., "anime with cool fights and a good story").
            n_results (int): The number of similar animes to retrieve from the vector store.

        Returns:
            str: The AI-generated recommendation text.
        """
        # 1. Create an embedding from the user's query
        print("Step 1/4: Generating embedding for the user query...")
        query_embedding = self.embedding_model.get_embedding(user_query)

        # 2. Query the vector store for similar animes
        print(f"Step 2/4: Querying vector store for {n_results} similar animes...")
        similar_animes: List[Dict[str, Any]] = self.vector_store.query_animes(
            query_embedding=query_embedding,
            n_results=n_results
        )

        if not similar_animes:
            return "I'm sorry, I couldn't find any animes that match your query. Please try being a bit more descriptive!"

        # 3. Create a prompt using the retrieved animes
        print("Step 3/4: Creating prompt for the language model...")
        prompt = self.prompt_template.create_recommendation_prompt(
            user_query=user_query,
            similar_animes=similar_animes
        )

        # 4. Get the final recommendation from the Groq model
        print("Step 4/4: Generating recommendation with Groq LLM...")
        recommendation = self.groq_model.generate_recommendation(prompt)
        print("Recommendation generated successfully.")
        
        return recommendation

# Singleton instance to be used across the application
recommender = Recommender()

