from typing import List, Dict, Any

class PromptTemplate:
    """
    Manages the creation of the final prompt to be sent to the LLM.
    
    This class takes the user's query and the context (similar animes)
    and formats them into a structured prompt with system instructions.
    """

    def __init__(self):
        self.system_instruction = (
            "You are an expert anime recommender. Your task is to provide a compelling, "
            "paragraph-style recommendation for a user based on their query and a list of "
            "potentially relevant anime. Analyze the provided anime details (genres, rating) "
            "to understand the user's taste.\n\n"
            "Your response should:\n"
            "1. Be a single, fluid paragraph, not a list.\n"
            "2. Recommend multiple suitable animes from the provided context (up to 10).\n"
            "3. Explain WHY you are recommending them, connecting their themes or genres "
            "to the user's query.\n"
            "4. Be engaging, friendly, and encouraging."
        )

    def format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Formats the list of similar animes into a string for the prompt.
        
        Args:
            context (List[Dict[str, Any]]): A list of anime data dictionaries from the vector store.
        
        Returns:
            str: A formatted string containing the details of the context animes.
        """
        formatted_contexts = []
        for anime in context:
            genre_list = anime.get('genre', [])
            genres = ", ".join(genre_list) if genre_list else "N/A"
            
            # --- THIS IS THE UPDATED PART ---
            # The 'Synopsis' field has been completely removed.
            context_str = (
                f"Name: {anime.get('name', 'N/A')}\n"
                f"Genres: {genres}\n"
                f"Type: {anime.get('type', 'N/A')}\n"
                f"Rating: {anime.get('rating', 'N/A')}"
            )
            formatted_contexts.append(context_str)
        return "\n---\n".join(formatted_contexts)

    def create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Creates the final, complete prompt string.
        
        Args:
            query (str): The user's original query.
            context (List[Dict[str, Any]]): The context animes from the vector store.
        
        Returns:
            str: The final prompt ready to be sent to the LLM.
        """
        formatted_context = self.format_context(context)
        
        return (
            f"{self.system_instruction}\n\n"
            f"--- USER QUERY ---\n"
            f"{query}\n\n"
            f"--- RELEVANT ANIME CONTEXT ---\n"
            f"{formatted_context}\n\n"
            f"--- RECOMMENDATION ---\n"
        )

# Create a singleton instance to be used across the application
PROMPT_TEMPLATE = PromptTemplate()

