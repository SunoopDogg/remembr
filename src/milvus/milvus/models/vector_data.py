"""VectorData model - computed vector embeddings for Milvus storage."""

from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from .caption_data import CaptionData


@dataclass(frozen=True, slots=True)
class VectorData:
    """Computed vector embeddings for Milvus storage."""
    caption_embedding: list[float]  # 1024-dim
    position_vector: list[float]    # 3-dim [x, y, z]
    time_vector: list[float]        # 2-dim [normalized_sec, fractional]
    theta: float                    # scalar orientation

    @classmethod
    def from_caption_data(
        cls,
        caption_data: CaptionData,
        embedding_model: SentenceTransformer
    ) -> 'VectorData':
        """Transform CaptionData into vector embeddings."""
        # Generate caption embedding
        caption_embedding = embedding_model.encode(
            caption_data.caption,
            normalize_embeddings=True,
            convert_to_tensor=False
        ).tolist()

        # Position vector (3D)
        position_vector = [
            caption_data.position_x,
            caption_data.position_y,
            caption_data.position_z
        ]

        # Time vector (2D: normalized seconds, fractional nanoseconds)
        timestamp_seconds = float(caption_data.timestamp_sec)
        timestamp_fractional = float(caption_data.timestamp_nanosec) * 1e-9
        time_vector = [
            timestamp_seconds / 1e9,  # Normalize to billions
            timestamp_fractional
        ]

        return cls(
            caption_embedding=caption_embedding,
            position_vector=position_vector,
            time_vector=time_vector,
            theta=caption_data.theta
        )

    def __post_init__(self):
        """Validate vector dimensions match Milvus schema."""
        if len(self.caption_embedding) != 1024:
            raise ValueError(
                f"Caption embedding must be 1024-dim, got {len(self.caption_embedding)}")
        if len(self.position_vector) != 3:
            raise ValueError(f"Position vector must be 3-dim, got {len(self.position_vector)}")
        if len(self.time_vector) != 2:
            raise ValueError(f"Time vector must be 2-dim, got {len(self.time_vector)}")
