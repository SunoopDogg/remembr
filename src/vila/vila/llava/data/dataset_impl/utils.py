import copy
import glob
import os
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from vila.llava.constants import DEFAULT_IMAGE_TOKEN
from vila.llava.data.base import BaseDataset
from vila.llava.media import Image, Video
from vila.llava.mm_utils import dynamic_process_images_and_prompt, process_images
from vila.llava.train.args import DataArguments
from vila.llava.utils import io, make_list
from vila.llava.utils.logging import logger
from vila.llava.utils.media import extract_media
from vila.llava.utils.tokenizer import preprocess_conversation


def _process_image(image: List[Any], data_args: DataArguments) -> torch.Tensor:
    return process_images(image, data_args.image_processor, data_args)


def _remove_media_tokens(text: str) -> str:
    for token in ["<image>", "<video>"]:
        text = text.replace(token + "\n", "").replace("\n" + token, "").replace(token, "")
    return text.strip()
