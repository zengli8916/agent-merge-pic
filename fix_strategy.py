from config import FIX_PROMPT_TEMPLATE, PASS_THRESHOLD
from models import Evaluation


def generate_fix_instruction(evaluation: Evaluation) -> str:
    issues_text = "\n".join(f"- {issue}" for issue in evaluation.issues)

    fix_parts = []
    scores = evaluation.scores

    if scores.pose_realism < PASS_THRESHOLD:
        fix_parts.append("- 姿势需要更自然，像真人一样。避免僵硬或不自然的姿态。请重点参考参考图中主播的姿势。")

    if scores.product_accuracy < PASS_THRESHOLD:
        fix_parts.append("- 产品外观必须严格匹配产品参考图。特别注意产品上的文字、logo、颜色要保持一致。不要幻想出产品图中没有的物品。")

    if scores.product_size < PASS_THRESHOLD:
        fix_parts.append("- 产品大小需要调整，使其与人物/身体的比例更加合理自然。")

    if scores.requirements_match < PASS_THRESHOLD:
        fix_parts.append("- 画面需要更符合用户的需求描述，请调整人物形象、背景环境等使其匹配。")

    if scores.physics < PASS_THRESHOLD:
        fix_parts.append("- 产品必须符合物理规律：不能悬浮、穿模。产品只展示外包装原始状态，不要幻想出内包装或打开的形态。")

    fix_instructions = "\n".join(fix_parts) if fix_parts else "- 整体质量需要提升，请生成更自然真实的图片。"

    return FIX_PROMPT_TEMPLATE.format(issues=issues_text, fix_instructions=fix_instructions)
