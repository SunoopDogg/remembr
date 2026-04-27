import datetime
from time import localtime, mktime, strftime
from typing import Tuple

from .qdrant_service import QdrantService, TEXT_VECTOR, POSITION_VECTOR, TIME_VECTOR
from .embedding_service import EmbeddingService
from ..config import TIMESTAMP_NORMALIZATION_EPOCH
from ..utils.protocols import Logger


class SearchService:
    """Vector similarity search service."""

    def __init__(
        self,
        qdrant_service: QdrantService,
        embedding_service: EmbeddingService,
        logger: Logger,
    ) -> None:
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        self.logger = logger

    def _execute_search(self, data: list, using: str, limit: int) -> list[dict]:
        results = self.qdrant_service.search(vector=data, using=using, limit=limit)
        return self._process_search_results(results)

    def search_by_text(self, query: str, limit: int = 10) -> list[dict]:
        try:
            query_embedding = self.embedding_service.encode(query)
            return self._execute_search(query_embedding, TEXT_VECTOR, limit)
        except Exception as e:
            self.logger.error(f'Error in search_by_text: {e}')
            return []

    def search_by_position(self, position: Tuple, limit: int = 10) -> list[dict]:
        try:
            if len(position) != 3:
                raise ValueError(f'Position must be 3D tuple (x,y,z), got {position}')
            position_vec = [float(p) for p in position]
            return self._execute_search(position_vec, POSITION_VECTOR, limit)
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f'Error in search_by_position: {e}')
            return []

    def search_by_time(self, time_str: str, limit: int = 10) -> list[dict]:
        try:
            query_timestamp = self._parse_time_string(time_str.strip())
            return self._execute_search([query_timestamp, 0.0], TIME_VECTOR, limit)
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f'Error in search_by_time: {e}')
            return []

    def _parse_time_string(self, time_str: str) -> float:
        """Parse HH:MM:SS or MM/DD/YYYY HH:MM:SS and return normalized timestamp."""
        mdy_date = strftime('%m/%d/%Y', localtime(TIMESTAMP_NORMALIZATION_EPOCH))
        template = '%m/%d/%Y %H:%M:%S'

        try:
            dt = datetime.datetime.strptime(time_str, template)
        except ValueError:
            parts = time_str.split(':')
            if len(parts) != 3:
                raise ValueError(f'Time must be HH:MM:SS format, got {time_str}')
            dt = datetime.datetime.strptime(mdy_date + ' ' + time_str, template)

        return mktime(dt.timetuple()) - TIMESTAMP_NORMALIZATION_EPOCH

    def _process_search_results(self, results) -> list[dict]:
        processed = []
        for hit in results:
            payload = hit.payload or {}
            time_value = payload.get('time', [0.0, 0.0])
            if isinstance(time_value, list) and len(time_value) >= 1:
                time_value = time_value[0]

            processed.append({
                'text': payload.get('caption', ''),
                'position': payload.get('position', [0.0, 0.0, 0.0]),
                'orientation': payload.get('theta', 0.0),
                'time': time_value,
                'distance': hit.score,
            })
        return processed

    def format_results(self, results: list[dict], query_info: str = '') -> str:
        if not results:
            return f'No memories found for query: {query_info}'

        output = ''
        for idx, doc in enumerate(results):
            actual_timestamp = doc.get('time', 0.0) + TIMESTAMP_NORMALIZATION_EPOCH
            time_str = strftime('%Y-%m-%d %H:%M:%S', localtime(actual_timestamp))
            position = doc.get('position', [0.0, 0.0, 0.0])
            pos_str = f'[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]'

            output += (
                f"[Result {idx + 1}] (relevance_score: {doc.get('distance', 0.0):.4f})\n"
                f"  POSITION: {pos_str}\n"
                f"  ORIENTATION: {doc.get('orientation', 0.0):.3f} radians\n"
                f"  TIME: {time_str}\n"
                f"  DESCRIPTION: {doc.get('text', '')}\n\n"
            )

        return output.strip()
