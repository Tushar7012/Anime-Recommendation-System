import chromadb
import logging
from typing import List
from .data_models import Anime
from .llm_models import embedding_model

# Configure logging
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages the ChromaDB vector store for anime embeddings.
    """
    def __init__(self, path: str = "./chroma_db", collection_name: str = "anime_recommendations"):
        """
        Initializes the VectorStore client and collection.
        """
        try:
            self.client = chromadb.PersistentClient(path=path)
            self.collection_name = collection_name  # <-- Store collection_name as instance attribute
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"ChromaDB client initialized at path '{path}' and collection '{collection_name}' is ready.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB. Error: {e}")
            raise
    
    def recreate_collection(self):
        """
        Deletes the existing collection if it exists and creates a new one.
        """
        try:
            logger.info(f"Attempting to reset collection '{self.collection_name}'...")  # <-- Use self.collection_name
            self.client.delete_collection(name=self.collection_name)
            logger.info("Successfully deleted old collection.")
        except ValueError:
            logger.warning(f"Collection '{self.collection_name}' did not exist. A new one will be created.")
        except Exception as e:
            logger.error(f"An error occurred while trying to delete collection: {e}", exc_info=True)
        
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' is now ready for data population.")

    def add_animes(self, animes: List[Anime], embeddings: List[List[float]]):
        if not animes or not embeddings:
            logger.warning("add_animes called with empty animes or embeddings list.")
            return

        ids = [str(anime.anime_id) for anime in animes]
        metadatas = []
        for anime in animes:
            data = anime.dict()
            # Convert list fields to string format acceptable by ChromaDB
            if isinstance(data.get('genre'), list):
                data['genre'] = ', '.join(data['genre'])
            metadatas.append(data)



        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"Failed to add batch to ChromaDB: {e}", exc_info=True)
            raise

    def find_similar_animes(self, query_embedding: List[float], n_results: int = 10) -> List[dict]:
        if not query_embedding:
            logger.error("find_similar_animes called with an empty query_embedding.")
            return []
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results.get('metadatas', [[]])[0]
        except Exception as e:
            logger.error(f"Failed to query ChromaDB: {e}", exc_info=True)
            return []

# Singleton instance to be used across the application
VECTOR_STORE = VectorStore()
