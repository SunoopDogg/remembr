from dataclasses import dataclass
from typing import Tuple, List
from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageSegment:
    """Immutable image segment data for processing."""

    images: Tuple[Image.Image, ...]
    timestamps: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.images) != len(self.timestamps):
            raise ValueError(
                f"Images ({len(self.images)}) and timestamps ({len(self.timestamps)}) "
                "must have the same length"
            )

    @classmethod
    def from_lists(
        cls,
        images: List[Image.Image],
        timestamps: List[float],
    ) -> 'ImageSegment':
        """Create segment from mutable lists."""
        return cls(tuple(images), tuple(timestamps))

    @property
    def image_count(self) -> int:
        """Number of images in the segment."""
        return len(self.images)

    @property
    def time_range(self) -> Tuple[float, float]:
        """Return (start_time, end_time) of the segment."""
        return (self.timestamps[0], self.timestamps[-1])

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.timestamps[-1] - self.timestamps[0]
