from dataclasses import dataclass
from typing import Optional

from ..utils.type_utils import safe_float, safe_str


@dataclass(frozen=True, slots=True)
class AgentOutput:
    """Structured output from agent query."""

    type: Optional[str] = None
    text: Optional[str] = None
    binary: Optional[str] = None
    position: Optional[tuple] = None
    orientation: Optional[float] = None
    time: Optional[float] = None
    duration: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'AgentOutput':
        """Create AgentOutput from dictionary."""
        # Extract orientation as single float (euler Z angle)
        orientation = d.get('orientation')
        if isinstance(orientation, (list, tuple)) and len(orientation) > 0:
            orientation = safe_float(orientation[0])
        else:
            orientation = safe_float(orientation)

        # Handle position - normalize 'null' to None
        position = d.get('position')
        if position == 'null' or position == '':
            position = None

        return cls(
            type=safe_str(d.get('type')),
            text=safe_str(d.get('text')),
            binary=safe_str(d.get('binary')),
            position=position,
            orientation=orientation,
            time=safe_float(d.get('time')),
            duration=safe_float(d.get('duration')),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type,
            'text': self.text,
            'binary': self.binary,
            'position': self.position,
            'orientation': self.orientation,
            'time': self.time,
            'duration': self.duration,
        }
