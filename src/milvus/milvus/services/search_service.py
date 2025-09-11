import datetime
import time
from time import localtime, strftime
from typing import Protocol, Tuple

from .milvus_service import MilvusService
from .embedding_service import EmbeddingService
from ..config import FIXED_SUBTRACT


class Logger(Protocol):
    """Protocol for ROS2-compatible logger."""

    def info(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


class SearchService:
    """Vector similarity search service."""

    def __init__(
        self,
        milvus_service: MilvusService,
        embedding_service: EmbeddingService,
        logger: Logger,
    ) -> None:
        self.milvus_service = milvus_service
        self.embedding_service = embedding_service
        self.logger = logger

    def search_by_text(self, query: str, limit: int = 5) -> list[dict]:
        """Search memories by text using vector similarity."""
        try:
            query_embedding = self.embedding_service.encode(query)

            results = self.milvus_service.client.search(
                collection_name=self.milvus_service.config.collection_name,
                data=[query_embedding],
                anns_field="caption_embedding",
                limit=limit,
                output_fields=["caption_text", "position", "theta", "time", "duration"]
            )

            return self._process_results(results)

        except Exception as e:
            self.logger.error(f'Error in search_by_text: {e}')
            return []

    def search_by_position(self, position: Tuple, limit: int = 5) -> list[dict]:
        """Search memories by spatial position."""
        try:
            if len(position) != 3:
                raise ValueError(f"Position must be 3D tuple (x,y,z), got {position}")

            position_vec = [float(p) for p in position]

            results = self.milvus_service.client.search(
                collection_name=self.milvus_service.config.collection_name,
                data=[position_vec],
                anns_field="position",
                limit=limit,
                output_fields=["caption_text", "position", "theta", "time", "duration"]
            )

            return self._process_results(results)

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f'Error in search_by_position: {e}')
            return []

    def search_by_time(self, time_str: str, limit: int = 5) -> list[dict]:
        """Search memories by time in HH:MM:SS format."""
        try:
            time_str = time_str.strip()

            # Get date from FIXED_SUBTRACT (same logic as remembr)
            t = localtime(FIXED_SUBTRACT)
            mdy_date = strftime('%m/%d/%Y', t)
            template = "%m/%d/%Y %H:%M:%S"

            # Check if already in full datetime format
            try:
                datetime.datetime.strptime(time_str, template)
                full_datetime = time_str
            except ValueError:
                # Assume HH:MM:SS format, add date
                time_parts = time_str.split(':')
                if len(time_parts) != 3:
                    raise ValueError(f"Time must be HH:MM:SS format, got {time_str}")
                full_datetime = mdy_date + ' ' + time_str

            # Convert to timestamp and subtract offset (remembr-compatible)
            query_timestamp = time.mktime(
                datetime.datetime.strptime(full_datetime, template).timetuple()
            ) - FIXED_SUBTRACT

            # Time vector format: [(timestamp - FIXED_SUBTRACT), 0.0]
            time_vec = [query_timestamp, 0.0]

            results = self.milvus_service.client.search(
                collection_name=self.milvus_service.config.collection_name,
                data=[time_vec],
                anns_field="time",
                limit=limit,
                output_fields=["caption_text", "position", "theta", "time", "duration"]
            )

            return self._process_results(results)

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f'Error in search_by_time: {e}')
            return []

    def _process_results(self, results) -> list[dict]:
        """Process Milvus search results into standardized format."""
        processed = []
        if results and len(results) > 0:
            for hit in results[0]:
                processed.append({
                    'caption_text': hit.get('entity', {}).get('caption_text', ''),
                    'position': hit.get('entity', {}).get('position', [0.0, 0.0, 0.0]),
                    'theta': hit.get('entity', {}).get('theta', 0.0),
                    'time': hit.get('entity', {}).get('time', [0.0, 0.0]),
                    'duration': hit.get('entity', {}).get('duration', 0.0),
                    'distance': hit.get('distance', 0.0)
                })
        return processed

    def format_results(self, results: list[dict], query_info: str = "") -> str:
        """Format search results into human-readable string."""
        if not results:
            return f"No memories found for query: {query_info}"

        output = ""
        for doc in results:
            time_vec = doc.get('time', [0.0, 0.0])

            # Convert from remembr format: time_vec[0] = (timestamp - FIXED_SUBTRACT)
            # Add FIXED_SUBTRACT back to get actual timestamp
            if len(time_vec) >= 1:
                actual_timestamp = time_vec[0] + FIXED_SUBTRACT
                t = localtime(actual_timestamp)
                time_str = strftime('%Y-%m-%d %H:%M:%S', t)
            else:
                time_str = "unknown"

            position = doc.get('position', [0.0, 0.0, 0.0])
            theta = doc.get('theta', 0.0)
            duration = doc.get('duration', 0.0)
            caption = doc.get('caption_text', '')

            output += (
                f"At time={time_str}, the robot was at an average position of "
                f"{[round(p, 3) for p in position]} facing theta={round(theta, 3)} rad "
                f"for {round(duration, 1)}s. "
                f"The robot saw the following: {caption}\n\n"
            )

        return output.strip()
