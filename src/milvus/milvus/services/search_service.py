import datetime
import time
from time import localtime, strftime
from typing import Tuple

from .milvus_service import MilvusService
from .embedding_service import EmbeddingService
from ..config import TIMESTAMP_NORMALIZATION_EPOCH
from ..utils.protocols import Logger


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

    def _execute_search(self, data: list, anns_field: str, limit: int) -> list[dict]:
        """Execute a vector search and return processed results."""
        results = self.milvus_service.search(data=[data], anns_field=anns_field, limit=limit)
        return self._process_search_results(results)

    def search_by_text(self, query: str, limit: int = 5) -> list[dict]:
        """Search memories by text using vector similarity."""
        try:
            query_embedding = self.embedding_service.encode(query)
            return self._execute_search(query_embedding, "text_embedding", limit)
        except Exception as e:
            self.logger.error(f'Error in search_by_text: {e}')
            return []

    def search_by_position(self, position: Tuple, limit: int = 5) -> list[dict]:
        """Search memories by spatial position."""
        try:
            if len(position) != 3:
                raise ValueError(f"Position must be 3D tuple (x,y,z), got {position}")
            position_vec = [float(p) for p in position]
            return self._execute_search(position_vec, "position", limit)
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f'Error in search_by_position: {e}')
            return []

    def search_by_time(self, time_str: str, limit: int = 5) -> list[dict]:
        """Search memories by time using vector similarity."""
        try:
            time_str = time_str.strip()
            query_timestamp = self._parse_time_string(time_str)
            time_vector = [query_timestamp, 0.0]
            return self._execute_search(time_vector, "time", limit)
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f'Error in search_by_time: {e}')
            return []

    def _parse_time_string(self, time_str: str) -> float:
        """Parse time string and return normalized timestamp."""
        t = localtime(TIMESTAMP_NORMALIZATION_EPOCH)
        mdy_date = strftime('%m/%d/%Y', t)
        template = "%m/%d/%Y %H:%M:%S"

        try:
            datetime.datetime.strptime(time_str, template)
            full_datetime = time_str
        except ValueError:
            time_parts = time_str.split(':')
            if len(time_parts) != 3:
                raise ValueError(f"Time must be HH:MM:SS format, got {time_str}")
            full_datetime = mdy_date + ' ' + time_str

        actual_timestamp = time.mktime(
            datetime.datetime.strptime(full_datetime, template).timetuple()
        )
        return actual_timestamp - TIMESTAMP_NORMALIZATION_EPOCH

    def _process_search_results(self, results) -> list[dict]:
        """Process Milvus vector search results."""
        processed = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get('entity', {})
                time_value = entity.get('time', [0.0, 0.0])
                if isinstance(time_value, list) and len(time_value) >= 1:
                    time_value = time_value[0]

                processed.append({
                    'text': entity.get('caption', ''),
                    'position': entity.get('position', [0.0, 0.0, 0.0]),
                    'orientation': entity.get('theta', 0.0),
                    'time': time_value,
                    'distance': hit.get('distance', 0.0),
                })
        return processed

    def format_results(self, results: list[dict], query_info: str = "") -> str:
        """Format search results into human-readable string."""
        if not results:
            return f"No memories found for query: {query_info}"

        output = ""
        for idx, doc in enumerate(results):
            timestamp = doc.get('time', 0.0)
            actual_timestamp = timestamp + TIMESTAMP_NORMALIZATION_EPOCH
            t = localtime(actual_timestamp)
            time_str = strftime('%Y-%m-%d %H:%M:%S', t)

            position = doc.get('position', [0.0, 0.0, 0.0])
            orientation = doc.get('orientation', 0.0)
            text = doc.get('text', '')
            distance = doc.get('distance', 0.0)

            # Format position explicitly for LLM extraction
            pos_str = f"[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"

            output += (
                f"[Result {idx + 1}] (relevance_score: {distance:.4f})\n"
                f"  POSITION: {pos_str}\n"
                f"  ORIENTATION: {orientation:.3f} radians\n"
                f"  TIME: {time_str}\n"
                f"  DESCRIPTION: {text}\n\n"
            )

        return output.strip()
