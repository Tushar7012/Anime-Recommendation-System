import chromadb
import logging
from typing import List, Dict, Any
from .data_models import Anime
from .llm_models import embedding_model

# Configure logging
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages the ChromaDB vector store for anime embeddings.
    
    This class handles the creation of a persistent vector database,
    adding anime data and their embeddings, and querying for anime that are
    semantically similar to a user's query.
    """
    def __init__(self, path: str = "./chroma_db", collection_name: str = "anime_recommendations"):
        """
        Initializes the VectorStore client and collection.

        Args:
            path (str): The directory path to store the persistent ChromaDB data.
            collection_name (str): The name of the collection to store anime data.
        """
        try:
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"ChromaDB client initialized at path '{path}' and collection '{collection_name}' is ready.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB. Error: {e}")
            raise

    def add_anime_batch(self, animes: List[Anime]):
        """
        Adds a batch of anime to the vector store. It creates embeddings for each
        anime's synopsis and stores them along with metadata.

        Args:
            animes (List[Anime]): A list of Anime Pydantic objects.
        """
        if not animes:
            logger.warning("Attempted to add an empty list of animes to the vector store.")
            return

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        logger.info(f"Processing a batch of {len(animes)} animes for the vector store.")
        for anime in animes:
            try:
                # The document stored is the synopsis, which is used for embedding
                doc = anime.synopsis
                embedding = embedding_model.create_embedding(doc)
                
                if not embedding:
                    logger.warning(f"Could not generate embedding for anime ID {anime.anime_id}. Skipping.")
                    continue

                ids.append(str(anime.anime_id))
                embeddings.append(embedding)
                # Store all other anime data as metadata
                metadatas.append(anime.model_dump(exclude={'synopsis'}))
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing anime ID {anime.anime_id}: {e}")

        if not ids:
            logger.warning("No valid animes were processed in this batch.")
            return

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Successfully added {len(ids)} animes to the vector store.")
        except Exception as e:
            logger.error(f"Failed to add batch to ChromaDB collection. Error: {e}")
            raise

    def find_similar_animes(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Finds the most similar animes in the vector store based on a text query.

        Args:
            query_text (str): The user's text describing their anime preference.
            n_results (int): The number of similar animes to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                 contains the metadata and similarity score of a matched anime.
        """
        try:
            query_embedding = embedding_model.create_embedding(query_text)
            if not query_embedding:
                logger.error("Could not generate embedding for the query text.")
                return []
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Combine metadata and distances (scores) into a single list
            similar_animes = []
            if results and results['metadatas'] and results['distances']:
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                for i in range(len(metadatas)):
                    anime_data = metadatas[i]
                    anime_data['score'] = 1 - distances[i] # Convert distance to similarity score
                    similar_animes.append(anime_data)
            
            return similar_animes

        except Exception as e:
            logger.error(f"Failed to query for similar animes. Error: {e}")
            return []

# Singleton instance to be used across the application
vector_store = VectorStore()

