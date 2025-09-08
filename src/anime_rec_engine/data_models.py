from pydantic import BaseModel, Field
from typing import List

class Anime(BaseModel):
    """
    Pydantic data model representing a single anime.
    
    This model enforces the data types for each attribute of an anime,
    ensuring consistency and providing validation. The `synopsis` is the
    most critical field for generating embeddings.
    """
    anime_id: int = Field(..., description="The unique identifier for the anime.")
    name: str = Field(..., description="The official name of the anime.")
    genres: List[str] = Field(..., description="A list of genres associated with the anime.")
    anime_type: str = Field(..., description="The type of the anime (e.g., TV, Movie, OVA).")
    episodes: int = Field(..., description="The total number of episodes.")
    rating: float = Field(..., description="The average user rating, out of 10.")
    members: int = Field(..., description="The number of community members who have this anime in their list.")
    synopsis: str = Field(..., description="A detailed summary of the anime's plot.")

    class Config:
        """
        Pydantic model configuration.
        `from_attributes` allows creating a model instance from an object's attributes.
        """
        from_attributes = True

