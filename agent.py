import io
import logging
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from config import MAX_RETRIES_BEFORE_RESTART, MAX_ROUNDS
from evaluator import evaluate
from fix_strategy import generate_fix_instruction
from generator import Conversation, continue_conversation, start_conversation
from models import AgentResult, Evaluation, Requirements
from prompt_builder import build_initial_prompt
from web import EventType, ProgressEvent

logger = logging.getLogger(__name__)


class LivestreamAgent:
    ASPECT_RATIOS = {
        "9:16": (1080, 1920),
        "3:4": (1080, 1440),
        "1:1": (1080, 1080),
        "4:3": (1440, 1080),
        "16:9": (1920, 1080),
    }

    def __init__(self, output_dir: str = "output", max_rounds: int | None = None, max_retries_before_restart: int | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds if max_rounds is not None else MAX_ROUNDS
        self.max_retries_before_restart = max_retries_before_restart if max_retries_before_restart is not None else MAX_RETRIES_BEFORE_RESTART
        self.aspect_ratio = "9:16"

    def run(self, reference_path: str, product_paths: list[str], requirements: Requirements, on_progress: Callable[[ProgressEvent], None] | None = None) -> AgentResult:
        reference_bytes = Path(reference_path).read_bytes()
        product_bytes_list = [Path(p).read_bytes() for p in product_paths]

        with Image.open(reference_path) as ref_img:
            ref_size = ref_img.size

        self.target_size = self._compute_target_size(ref_size)

        initial_prompt = build_initial_prompt(requirements, reference_path, self.target_size)
        logger.info(f"Starting initial generation... target size: {self.target_size}, products: {len(product_paths)}")
        self._emit(on_progress, EventType.STARTED, 0, {"target_size": list(self.target_size), "max_rounds": self.max_rounds, "prompt": initial_prompt})

        image_paths = [reference_path] + product_paths
        conv = start_conversation(initial_prompt, image_paths)
        current_image = conv.latest_image
        self._save_intermediate(current_image, 0)
        logger.info("Round 0: initial image generated")
        self._emit(on_progress, EventType.ROUND_IMAGE, 0, {"image": "round_0.png", "prompt": initial_prompt})

        best_image = current_image
        best_score = 0.0
        best_conv = conv
        prev_score = 0.0
        consecutive_failures = 0
        last_image_round = 0

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Round {round_num}: evaluating...")
            self._emit(on_progress, EventType.EVALUATING, last_image_round, {})

            evaluation = evaluate(current_image, reference_bytes, product_bytes_list, requirements)
            current_score = evaluation.total_score
            logger.info(f"Round {round_num}: scores={evaluation.scores.model_dump()}, total={current_score:.1f}, passed={evaluation.passed}")
            self._emit(on_progress, EventType.EVALUATION_DONE, last_image_round, {
                "scores": evaluation.scores.model_dump(),
                "total": round(current_score, 1),
                "issues": evaluation.issues,
                "passed": evaluation.passed,
            })

            if evaluation.passed:
                final_path = self._save_final(current_image)
                logger.info(f"Passed at round {round_num}!")
                self._emit(on_progress, EventType.COMPLETED, round_num, {"image": "final.png", "partial": False, "scores": evaluation.scores.model_dump()})
                return AgentResult(image_path=str(final_path), rounds=round_num, evaluation=evaluation)

            # Good enough: high total score with no critical failures (all >= 3)
            min_score = min(evaluation.scores.model_dump().values())
            if current_score >= 4.5 and min_score >= 3:
                final_path = self._save_final(current_image)
                logger.info(f"Good enough at round {round_num} (total={current_score:.1f}, min={min_score})")
                self._emit(on_progress, EventType.COMPLETED, round_num, {"image": "final.png", "partial": False, "scores": evaluation.scores.model_dump()})
                return AgentResult(image_path=str(final_path), rounds=round_num, evaluation=evaluation)

            if current_score > best_score:
                best_score = current_score
                best_image = current_image
                best_conv = conv

            # Score dropped: rollback to best state and restart from there
            if prev_score > 0 and current_score < prev_score - 0.3:
                logger.info(f"Round {round_num}: score dropped ({prev_score:.1f} -> {current_score:.1f}), rolling back to best state")
                self._emit(on_progress, EventType.RESTARTING, round_num, {"reason": f"分数下降 {prev_score:.1f} → {current_score:.1f}，回滚到最佳状态重新生成"})
                consecutive_failures = 0
                conv = best_conv
                current_image = best_image
                try:
                    fix_instruction = generate_fix_instruction(evaluation)
                    conv = start_conversation(initial_prompt, image_paths)
                    current_image = conv.latest_image
                    self._save_intermediate(current_image, round_num)
                    last_image_round = round_num
                    self._emit(on_progress, EventType.ROUND_IMAGE, round_num, {"image": f"round_{round_num}.png", "prompt": initial_prompt + "\n[回滚重新生成]"})
                except Exception as e:
                    logger.warning(f"Round {round_num}: rollback restart failed: {e}")
                prev_score = 0.0
                continue

            consecutive_failures += 1

            if consecutive_failures >= self.max_retries_before_restart:
                logger.info(f"Round {round_num}: {consecutive_failures} consecutive failures, restarting fresh...")
                self._emit(on_progress, EventType.RESTARTING, round_num, {"reason": f"连续 {consecutive_failures} 轮未通过，重新生成"})
                consecutive_failures = 0
                try:
                    conv = start_conversation(initial_prompt, image_paths)
                    current_image = conv.latest_image
                    self._save_intermediate(current_image, round_num)
                    last_image_round = round_num
                    self._emit(on_progress, EventType.ROUND_IMAGE, round_num, {"image": f"round_{round_num}.png", "prompt": initial_prompt + "\n[重新生成]"})
                except Exception as e:
                    logger.warning(f"Round {round_num}: fresh restart failed: {e}")
                prev_score = 0.0
                continue

            fix_instruction = generate_fix_instruction(evaluation)
            self._emit(on_progress, EventType.FIXING, round_num, {"prompt": fix_instruction})
            logger.info(f"Round {round_num}: applying fix...")

            try:
                conv = continue_conversation(conv, fix_instruction)
                current_image = conv.latest_image
                self._save_intermediate(current_image, round_num)
                last_image_round = round_num
                self._emit(on_progress, EventType.ROUND_IMAGE, round_num, {"image": f"round_{round_num}.png", "prompt": fix_instruction})
            except Exception as e:
                logger.warning(f"Round {round_num}: fix generation failed: {e}")
                continue

            prev_score = current_score

        final_path = self._save_final(best_image)
        logger.info(f"Max rounds reached. Best result saved to {final_path}")
        self._emit(on_progress, EventType.COMPLETED, self.max_rounds, {"image": "final.png", "partial": True})
        return AgentResult(image_path=str(final_path), rounds=self.max_rounds, partial=True)

    def _emit(self, callback: Callable | None, event_type: EventType, round_num: int, data: dict):
        if callback:
            callback(ProgressEvent(event=event_type, round=round_num, data=data))

    def _compute_target_size(self, ref_size: tuple[int, int]) -> tuple[int, int]:
        if self.aspect_ratio == "auto":
            return ref_size
        return self.ASPECT_RATIOS.get(self.aspect_ratio, (1080, 1920))

    def _resize_to_target(self, image_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(image_bytes))
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _save_intermediate(self, image_bytes: bytes, round_num: int) -> Path:
        path = self.output_dir / f"round_{round_num}.png"
        path.write_bytes(image_bytes)
        return path

    def _save_final(self, image_bytes: bytes) -> Path:
        resized = self._resize_to_target(image_bytes)
        path = self.output_dir / "final.png"
        path.write_bytes(resized)
        return path
