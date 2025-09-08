import logging
from sentence_transformers import SentenceTransformer
from groq import Groq
from .config import config

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    A wrapper class for the SentenceTransformer model from Hugging Face.
    
    This class handles loading the embedding model and provides a simple
    interface to convert text (anime synopses) into vector embeddings.
    """
    def __init__(self, model_name: str = config.embedding_model_name):
        """
        Initializes the EmbeddingModel.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}'. Error: {e}")
            raise

    def create_embedding(self, text: str) -> list[float]:
        """
        Generates a vector embedding for the given text.

        Args:
            text (str): The input text (e.g., an anime synopsis).

        Returns:
            list[float]: The generated vector embedding.
        """
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to create embedding. Error: {e}")
            return []

class GroqModel:
    """
    A wrapper class for the Groq API.

    This class handles communication with the Groq large language model (LLM)
    to generate human-readable anime recommendations based on a given prompt.
    """
    def __init__(self, api_key: str = config.groq_api_key, model_name: str = config.llm_model_name):
        """
        Initializes the GroqModel client.

        Args:
            api_key (str): The API key for the Groq service.
            model_name (str): The name of the LLM to use (e.g., 'llama3-8b-8192').
        """
        if not api_key:
            raise ValueError("Groq API key is missing.")
        try:
            self.client = Groq(api_key=api_key)
            self.model_name = model_name
            logger.info("Groq client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client. Error: {e}")
            raise

    def get_recommendation(self, prompt: str) -> str:
        """
        Generates a recommendation by sending a prompt to the Groq LLM.

        Args:
            prompt (str): The complete prompt for the LLM.

        Returns:
            str: The text content of the LLM's response.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
            )
            response = chat_completion.choices[0].message.content
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to get recommendation from Groq. Error: {e}")
            return "Sorry, I was unable to generate a recommendation at this time."

# Create singleton instances to be used across the application
embedding_model = EmbeddingModel()
groq_model = GroqModel()

