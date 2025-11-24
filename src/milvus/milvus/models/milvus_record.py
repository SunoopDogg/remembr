import time
from dataclasses import dataclass

from .caption_data import CaptionData
from .vector_data import VectorData


@dataclass(frozen=True, slots=True)
class MilvusMemoryRecord:
    """Record format for Milvus database insertion."""
    id: str
    caption_text: str
    caption_embedding: list[float]
    position: list[float]
    theta: float
    time: list[float]

    @classmethod
    def from_caption_and_vectors(
        cls,
        caption_data: CaptionData,
        vector_data: VectorData
    ) -> 'MilvusMemoryRecord':
        """Create Milvus record from caption and vector data."""
        return cls(
            id=str(time.time()),  # Fixed bug: was time.time_module.time()
            caption_text=caption_data.caption,
            caption_embedding=vector_data.caption_embedding,
            position=vector_data.position_vector,
            theta=vector_data.theta,
            time=vector_data.time_vector
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for Milvus insertion."""
        return {
            "id": self.id,
            "caption_text": self.caption_text,
            "caption_embedding": self.caption_embedding,
            "position": self.position,
            "theta": self.theta,
            "time": self.time
        }

    def __post_init__(self):
        """Validate record matches Milvus schema."""
        if not self.id:
            raise ValueError("ID cannot be empty")
        if len(self.id) > 1000:
            raise ValueError("ID exceeds max length of 1000")
        if len(self.caption_text) > 3000:
            raise ValueError("Caption text exceeds max length of 3000")
