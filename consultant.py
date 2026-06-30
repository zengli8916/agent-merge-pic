import logging
from pathlib import Path

from config import DESIGN_CONSULTATION_PROMPT
from generator import call_vision
from models import Requirements

logger = logging.getLogger(__name__)


def run_consultation(reference_path: str, product_paths: list[str], requirements: Requirements) -> str:
    existing_info = ""
    if requirements.description:
        existing_info = f"用户已提供的需求信息：\n{requirements.description}"

    prompt = DESIGN_CONSULTATION_PROMPT.format(existing_info=existing_info)

    image_list = [Path(reference_path).read_bytes()]
    for p in product_paths:
        image_list.append(Path(p).read_bytes())

    response = call_vision(prompt, image_list)
    return response
