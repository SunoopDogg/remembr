import os
from typing import List, Protocol

from PIL import Image
from transformers import GenerationConfig

from vila import llava
from vila.llava import conversation as clib
from vila.llava.utils.logging import logger as vila_logger


class Logger(Protocol):
    """Protocol for ROS2-compatible logger."""

    def info(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...
    def fatal(self, msg: str) -> None: ...


class VilaService:
    """VILA model management and inference service."""

    def __init__(
        self,
        model_path: str,
        conv_mode: str,
        logger: Logger,
    ) -> None:
        self._model_path = model_path
        self._conv_mode = conv_mode
        self._logger = logger
        self._model = None

    def load_model(self) -> None:
        """Load VILA model with error handling."""
        vila_logger.info(f"Loading VILA model from: {self._model_path}")

        try:
            self._model = llava.load(self._model_path, model_base=None)
            vila_logger.info("VILA model loaded successfully")
        except FileNotFoundError as e:
            self._logger.fatal(f"Model path not found: {self._model_path}")
            raise RuntimeError(f"Failed to load VILA model: {e}") from e
        except Exception as e:
            self._logger.fatal(f"Failed to load VILA model: {e}")
            raise RuntimeError(f"Failed to load VILA model: {e}") from e

        # Configure PS3 and context length
        self._configure_ps3_and_context_length()

        # Set conversation mode
        clib.default_conversation = clib.conv_templates[self._conv_mode].copy()
        vila_logger.info(f"Using conversation mode: {self._conv_mode}")

    def _configure_ps3_and_context_length(self) -> None:
        """Configure PS3 settings and adjust context length from environment variables."""
        model = self._model

        # Get PS3 configs from environment variables
        num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
        num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
        select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
        look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
        smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

        # Set PS3 configs
        if num_look_close is not None:
            vila_logger.info(f"Num look close: {num_look_close}")
            model.num_look_close = int(num_look_close)
        if num_token_look_close is not None:
            vila_logger.info(f"Num token look close: {num_token_look_close}")
            model.num_token_look_close = int(num_token_look_close)
        if select_num_each_scale is not None:
            vila_logger.info(f"Select num each scale: {select_num_each_scale}")
            select_num_each_scale = [
                int(x) for x in select_num_each_scale.split("+")
            ]
            model.get_vision_tower().vision_tower.vision_model.max_select_num_each_scale = (
                select_num_each_scale
            )
        if look_close_mode is not None:
            vila_logger.info(f"Look close mode: {look_close_mode}")
            model.look_close_mode = look_close_mode
        if smooth_selection_prob is not None:
            vila_logger.info(f"Smooth selection prob: {smooth_selection_prob}")
            if smooth_selection_prob.lower() == "true":
                smooth_selection_prob = True
            elif smooth_selection_prob.lower() == "false":
                smooth_selection_prob = False
            else:
                raise ValueError(
                    f"Invalid smooth selection prob: {smooth_selection_prob}"
                )
            model.smooth_selection_prob = smooth_selection_prob

        # Adjust max context length based on PS3 config
        context_length = model.tokenizer.model_max_length
        if num_look_close is not None:
            context_length = max(
                context_length, int(num_look_close) * 2560 // 4 + 1024
            )
        if num_token_look_close is not None:
            context_length = max(
                context_length, int(num_token_look_close) // 4 + 1024
            )
        context_length = max(
            getattr(model.tokenizer, "model_max_length", context_length),
            context_length,
        )
        model.config.model_max_length = context_length
        model.config.tokenizer_model_max_length = context_length
        model.llm.config.model_max_length = context_length
        model.llm.config.tokenizer_model_max_length = context_length
        model.tokenizer.model_max_length = context_length

    def generate_caption(
        self,
        images: List[Image.Image],
        prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate caption for image sequence."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build prompt: [Image1, Image2, ..., ImageN, prompt_text]
        content = images.copy()
        content.append(prompt)

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        return self._model.generate_content(content, generation_config=generation_config)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def cleanup(self) -> None:
        """Release model resources."""
        self._model = None
