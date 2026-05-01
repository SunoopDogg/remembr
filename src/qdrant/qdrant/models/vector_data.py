from dataclasses import dataclass

from .caption_data import CaptionData
from ..config import TIMESTAMP_NORMALIZATION_EPOCH


@dataclass(frozen=True, slots=True)
class VectorData:
    """Computed vector embeddings for Qdrant storage."""

    text_embedding: list[float]
    position_vector: list[float]
    time: list[float]  # 2D vector: [normalized_time, 0.0]
    theta: float

    @classmethod
    def from_caption_data(
        cls,
        caption_data: CaptionData,
        text_embedding: list[float],
    ) -> 'VectorData':
        """Create VectorData from CaptionData and a pre-computed text embedding."""
        position_vector = [
            caption_data.position_x,
            caption_data.position_y,
            caption_data.position_z,
        ]

        nanosec_offset = caption_data.timestamp_nanosec * 1e-9
        timestamp_seconds = float(caption_data.timestamp_sec) + nanosec_offset
        normalized_time = timestamp_seconds - TIMESTAMP_NORMALIZATION_EPOCH

        return cls(
            text_embedding=text_embedding,
            position_vector=position_vector,
            time=[normalized_time, 0.0],
            theta=caption_data.theta,
        )

    def __post_init__(self) -> None:
        if not self.text_embedding:
            raise ValueError("Text embedding cannot be empty")
        if len(self.position_vector) != 3:
            raise ValueError(f"Position vector must be 3-dim, got {len(self.position_vector)}")
        if len(self.time) != 2:
            raise ValueError(f"Time vector must be 2-dim, got {len(self.time)}")
