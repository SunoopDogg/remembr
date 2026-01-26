from dataclasses import dataclass
from typing import Union

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from .caption_data import CaptionData
from ..config import FIXED_SUBTRACT


@dataclass(frozen=True, slots=True)
class VectorData:
    """Computed vector embeddings for Milvus storage."""

    caption_embedding: list[float]
    position_vector: list[float]
    time_vector: list[float]
    theta: float

    @classmethod
    def from_caption_data(
        cls,
        caption_data: CaptionData,
        embedding_model: Union[SentenceTransformer, HuggingFaceEmbeddings],
    ) -> 'VectorData':
        """Transform CaptionData into vector embeddings."""
        if isinstance(embedding_model, HuggingFaceEmbeddings):
            caption_embedding = embedding_model.embed_query(caption_data.caption)
        else:
            caption_embedding = embedding_model.encode(
                caption_data.caption,
                normalize_embeddings=True,
                convert_to_tensor=False
            ).tolist()

        position_vector = [
            caption_data.position_x,
            caption_data.position_y,
            caption_data.position_z
        ]

        timestamp_seconds = float(caption_data.timestamp_sec) + (caption_data.timestamp_nanosec * 1e-9)
        time_vector = [
            timestamp_seconds - FIXED_SUBTRACT,
            0.0
        ]

        return cls(
            caption_embedding=caption_embedding,
            position_vector=position_vector,
            time_vector=time_vector,
            theta=caption_data.theta
        )

    def __post_init__(self) -> None:
        if len(self.caption_embedding) != 1024:
            raise ValueError(
                f"Caption embedding must be 1024-dim, got {len(self.caption_embedding)}")
        if len(self.position_vector) != 3:
            raise ValueError(f"Position vector must be 3-dim, got {len(self.position_vector)}")
        if len(self.time_vector) != 2:
            raise ValueError(f"Time vector must be 2-dim, got {len(self.time_vector)}")
