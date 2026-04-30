from dataclasses import dataclass
from typing import Tuple, List
from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageSegment:

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
        return cls(tuple(images), tuple(timestamps))

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def time_range(self) -> Tuple[float, float]:
        return (self.timestamps[0], self.timestamps[-1])
