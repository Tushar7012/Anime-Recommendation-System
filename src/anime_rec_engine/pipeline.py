import pandas as pd
from typing import List, Generator
import logging
from tqdm import tqdm

from anime_rec_engine.data_models import Anime
from anime_rec_engine.llm_models import embedding_model
from anime_rec_engine.vector_store import VECTOR_STORE  # <-- Import the instance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_anime_data(data_filepath: str) -> List[Anime]:
    logging.info(f"Loading data from {data_filepath}...")
    try:
        df = pd.read_csv(data_filepath)
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_filepath}")
        return []
    
    # Drop rows with critical missing data
    df.dropna(subset=['anime_id', 'name', 'genre'], inplace=True)
    
    # Fill missing ratings with a neutral value
    df['rating'].fillna(0.0, inplace=True)
    df['episodes'].fillna(0, inplace=True)  # <-- Fill NaN episodes with 0

    # Convert data types safely
    df['anime_id'] = pd.to_numeric(df['anime_id'], errors='coerce')
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce', downcast='integer').fillna(0).astype(int)  # <-- Safe coercion
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)
    df['members'] = pd.to_numeric(df['members'], errors='coerce', downcast='integer').fillna(0).astype(int)
    df.dropna(subset=['anime_id'], inplace=True)

    # Convert genre from string to list
    df['genre'] = df['genre'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])

    anime_list = []
    for _, row in df.iterrows():
        try:
            anime_list.append(Anime(**row.to_dict()))
        except Exception as e:
            logging.warning(f"Skipping row due to validation error: {e} | Data: {row.to_dict()}")

    logging.info(f"Successfully loaded and validated {len(anime_list)} anime records.")
    return anime_list

def batch_generator(data: List, batch_size: int) -> Generator[List, None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def run_training_pipeline(data_filepath: str, batch_size: int = 50):
    anime_data = load_anime_data(data_filepath)
    if not anime_data:
        logging.error("Pipeline stopped: No data loaded.")
        return

    logging.info("Starting to generate embeddings and populate the vector store...")
    
    # Reset the collection using the instance
    VECTOR_STORE.recreate_collection()

    total_batches = (len(anime_data) + batch_size - 1) // batch_size
    
    for batch in tqdm(batch_generator(anime_data, batch_size), total=total_batches, desc="Processing batches"):
        texts_to_embed = ["Genres: " + ", ".join(anime.genre) for anime in batch]
        embeddings = [embedding_model.create_embedding(text) for text in texts_to_embed]

        if embeddings:
            VECTOR_STORE.add_animes(animes=batch, embeddings=embeddings)
        else:
            logging.warning("Skipping batch due to embedding failure.")
            
    logging.info("Training pipeline completed successfully. Vector store is populated.")
