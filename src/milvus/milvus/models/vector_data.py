from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings

from .caption_data import CaptionData
from ..config import FIXED_SUBTRACT


@dataclass(frozen=True, slots=True)
class VectorData:
    """Computed vector embeddings for Milvus storage."""

    text_embedding: list[float]
    position_vector: list[float]
    time: list[float]  # 2D vector: [normalized_time, 0.0]
    theta: float

    @classmethod
    def from_caption_data(
        cls,
        caption_data: CaptionData,
        embedding_model: HuggingFaceEmbeddings,
    ) -> 'VectorData':
        """Create VectorData from CaptionData using embedding model."""
        text_embedding = embedding_model.embed_query(caption_data.caption)

        position_vector = [
            caption_data.position_x,
            caption_data.position_y,
            caption_data.position_z
        ]

        timestamp_seconds = float(caption_data.timestamp_sec) + (caption_data.timestamp_nanosec * 1e-9)
        normalized_time = timestamp_seconds - FIXED_SUBTRACT
        time_vector = [normalized_time, 0.0]

        return cls(
            text_embedding=text_embedding,
            position_vector=position_vector,
            time=time_vector,
            theta=caption_data.theta,
        )

    def __post_init__(self) -> None:
        if not self.text_embedding:
            raise ValueError("Text embedding cannot be empty")
        if len(self.position_vector) != 3:
            raise ValueError(f"Position vector must be 3-dim, got {len(self.position_vector)}")
        if len(self.time) != 2:
            raise ValueError(f"Time vector must be 2-dim, got {len(self.time)}")
