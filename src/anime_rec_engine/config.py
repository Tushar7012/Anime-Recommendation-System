import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file at the project root
load_dotenv()
logger.info("Loading environment variables from .env file.")

class AppConfig:
    """
    Application configuration class.
    Loads sensitive information and settings from environment variables.
    Provides default values for non-critical settings.
    """
    def __init__(self):
        # API Keys - These are critical for the application to function
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN") # Optional for some models

        # Application Settings with sensible defaults
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.llm_model_name = os.getenv("LLM_MODEL_NAME", "llama3-8b-8192")

        # --- Validation ---
        # Ensure the most critical API key is present
        if not self.groq_api_key:
            logger.error("FATAL: GROQ_API_KEY environment variable not set.")
            raise ValueError("GROQ_API_KEY must be set in your .env file.")
        
        logger.info("Configuration loaded successfully.")

# Create a single, importable instance of the configuration
config = AppConfig()

