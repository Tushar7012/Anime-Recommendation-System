from anime_rec_engine.pipeline import run_training_pipeline

if __name__ == "__main__":
    DATA_FILEPATH = "D:/LLMOps_ALOps/Anime Recommendation System/Anime-Recommendation-System/data/anime.csv"
    run_training_pipeline(data_filepath=DATA_FILEPATH)