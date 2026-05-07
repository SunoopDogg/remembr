import uuid
from dataclasses import dataclass

from qdrant_client.models import PointStruct

from .caption_data import CaptionData
from ..config import TIMESTAMP_NORMALIZATION_EPOCH


@dataclass(frozen=True, slots=True)
class QdrantMemoryRecord:
    """Immutable record for Qdrant database insertion."""

    id: str
    text_embedding: list[float]
    position: list[float]
    theta: float
    time: list[float]  # 2D vector: [normalized_time, 0.0]
    caption: str

    @classmethod
    def from_caption_data(
        cls,
        caption_data: CaptionData,
        text_embedding: list[float],
    ) -> 'QdrantMemoryRecord':
        position = [caption_data.position_x, caption_data.position_y, caption_data.position_z]
        normalized_time = (
            float(caption_data.timestamp_sec)
            + caption_data.timestamp_nanosec * 1e-9
            - TIMESTAMP_NORMALIZATION_EPOCH
        )
        return cls(
            id=str(uuid.uuid4()),
            text_embedding=text_embedding,
            position=position,
            theta=caption_data.theta,
            time=[normalized_time, 0.0],
            caption=caption_data.caption,
        )

    def to_point(self) -> PointStruct:
        """Convert to Qdrant PointStruct for upsert."""
        return PointStruct(
            id=self.id,
            vector={
                'text_embedding': self.text_embedding,
                'position':       self.position,
                'time':           self.time,
            },
            payload={
                'theta':    self.theta,
                'caption':  self.caption,
                'position': self.position,
                'time':     self.time,
            },
        )

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError('ID cannot be empty')
        if len(self.id) > 1000:
            raise ValueError('ID exceeds max length of 1000')
        if len(self.caption) > 3000:
            raise ValueError('Caption exceeds max length of 3000')
        if len(self.time) != 2:
            raise ValueError('Time must be 2D vector')
