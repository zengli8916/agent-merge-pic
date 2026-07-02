from PIL import Image

from config import INITIAL_PROMPT_TEMPLATE
from models import Requirements


def build_initial_prompt(requirements: Requirements, reference_path: str, target_size: tuple[int, int] | None = None) -> str:
    if target_size:
        width, height = target_size
    else:
        with Image.open(reference_path) as img:
            width, height = img.size

    return INITIAL_PROMPT_TEMPLATE.format(
        description=requirements.description,
        width=width,
        height=height,
    )
