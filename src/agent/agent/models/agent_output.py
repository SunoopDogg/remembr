from dataclasses import dataclass
from typing import Optional


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
            orientation = float(orientation[0])
        elif orientation is not None:
            orientation = float(orientation)

        return cls(
            type=d.get('type'),
            text=d.get('text'),
            binary=d.get('binary'),
            position=d.get('position'),
            orientation=orientation,
            time=d.get('time'),
            duration=d.get('duration')
        )
