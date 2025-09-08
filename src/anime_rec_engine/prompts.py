from typing import List, Dict, Any

class PromptTemplate:
    """
    Manages the creation of a detailed prompt for the Groq LLM.
    
    This class takes the user's query and a list of similar animes found
    in the vector store, and formats them into a structured prompt that
    guides the language model to generate a personalized and coherent
    recommendation.
    """

    def _format_anime_list(self, animes: List[Dict[str, Any]]) -> str:
        """
        Formats the list of similar animes into a readable string.

        Args:
            animes (List[Dict[str, Any]]): A list of anime data dictionaries.

        Returns:
            str: A formatted string containing the details of each anime.
        """
        if not animes:
            return "No similar animes found."
        
        formatted_str = ""
        for i, anime in enumerate(animes, 1):
            formatted_str += f"{i}. **{anime.get('name', 'N/A')}** (Rating: {anime.get('rating', 'N/A')}/10)\n"
            formatted_str += f"   - **Genres:** {', '.join(anime.get('genres', []))}\n"
            formatted_str += f"   - **Type:** {anime.get('anime_type', 'N/A')}, Episodes: {anime.get('episodes', 'N/A')}\n\n"
        return formatted_str.strip()

    def create_recommendation_prompt(self, user_query: str, similar_animes: List[Dict[str, Any]]) -> str:
        """
        Creates the full prompt to be sent to the Groq LLM.

        Args:
            user_query (str): The original query from the user.
            similar_animes (List[Dict[str, Any]]): The list of similar animes retrieved
                                                   from the vector store.

        Returns:
            str: The final, structured prompt for the LLM.
        """
        
        formatted_animes = self._format_anime_list(similar_animes)

        prompt = f"""
        **System Instruction:** You are an AI Anime Recommendation expert.
        Your task is to provide a personalized anime recommendation based on the user's request and a list of potentially similar animes.
        Analyze the user's query and the provided anime list.
        Your goal is to recommend multiple suitable animes from the provided list. You can recommend up to 10 if they are a good fit.
        Generate a friendly, engaging, and concise recommendation.
        Explain WHY you are recommending the animes, connecting your choices directly to the user's query.
        Structure your response clearly. You can highlight 2-3 top choices with more detail.

        ---

        **User's Request:**
        "{user_query}"

        ---

        **Here is a list of animes that are semantically similar to the user's request. Use this list as your primary source for the recommendation:**
        {formatted_animes}

        ---

        **Your Recommendations:**
        Based on your interest in "{user_query}", here are some animes you might enjoy:
        """
        
        return prompt

# Singleton instance to be used across the application
prompt_template = PromptTemplate()

