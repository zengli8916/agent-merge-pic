import json
import logging
import re

from config import EVALUATION_PROMPT_TEMPLATE, PASS_THRESHOLD
from generator import call_vision
from models import Evaluation, EvaluationScores, Requirements

logger = logging.getLogger(__name__)


def evaluate(
    generated_image: bytes,
    reference_image: bytes,
    product_images: list[bytes],
    requirements: Requirements,
) -> Evaluation:
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        description=requirements.description,
        threshold=PASS_THRESHOLD,
    )

    try:
        image_list = [generated_image, reference_image] + product_images
        text = call_vision(prompt, image_list)
        return _parse_evaluation(text)
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        return Evaluation(
            scores=EvaluationScores(pose_realism=2, product_accuracy=2, product_size=2, requirements_match=2, physics=2),
            issues=["评估调用失败，需要重新生成"],
            passed=False,
        )


def _parse_evaluation(text: str) -> Evaluation:
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        logger.warning(f"No JSON found in evaluation response: {text[:200]}")
        return Evaluation(issues=["评估响应格式错误"], passed=False)

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in evaluation: {json_match.group()[:200]}")
        return Evaluation(issues=["评估JSON解析失败"], passed=False)

    scores = EvaluationScores(**(data.get("scores", {})))
    issues = data.get("issues", [])
    passed = data.get("passed", False)

    all_above = all(
        v >= PASS_THRESHOLD
        for v in [scores.pose_realism, scores.product_accuracy, scores.product_size, scores.requirements_match, scores.physics]
    )
    passed = passed and all_above

    return Evaluation(scores=scores, issues=issues, passed=passed)
