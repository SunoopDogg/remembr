import base64
import io
import time
import traceback
from typing import List

import httpx
from PIL import Image

from ..utils.protocols import Logger


class GemmaService:

    def __init__(self, model_name: str, base_url: str, logger: Logger) -> None:
        self._model_name = model_name
        self._base_url = base_url
        self._logger = logger
        self._client: httpx.Client | None = None

    def load_model(self, max_retries: int = 30, retry_interval: float = 5.0) -> None:
        """Connect to vLLM server with retry."""
        self._client = httpx.Client(base_url=self._base_url, timeout=120.0)
        for attempt in range(1, max_retries + 1):
            try:
                self._logger.info(
                    f'Connecting to VLM server at {self._base_url} '
                    f'(attempt {attempt}/{max_retries})...'
                )
                self._client.get('/v1/models').raise_for_status()
                self._logger.info(f'VLM server ready (model={self._model_name})')
                return
            except Exception as e:
                if attempt == max_retries:
                    self._logger.error(
                        f'Failed to connect after {max_retries} attempts: {e}'
                    )
                    self._logger.error(traceback.format_exc())
                    raise
                self._logger.warning(
                    f'Server not ready ({e}), retrying in {retry_interval}s...'
                )
                time.sleep(retry_interval)

    @staticmethod
    def _pil_to_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'data:image/jpeg;base64,{b64}'

    def generate_caption(
        self,
        images: List[Image.Image],
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        if self._client is None:
            raise RuntimeError('VLM service not connected. Call load_model() first.')

        content = [
            {
                'type': 'image_url',
                'image_url': {'url': self._pil_to_data_url(img)},
            }
            for img in images
        ]
        content.append({'type': 'text', 'text': prompt})

        response = self._client.post('/v1/chat/completions', json={
            'model': self._model_name,
            'messages': [{'role': 'user', 'content': content}],
            'temperature': temperature,
            'max_tokens': max_tokens,
        })
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def cleanup(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._logger.info('VLM service connection closed')
