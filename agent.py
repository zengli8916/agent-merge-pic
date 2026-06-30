import io
import logging
from pathlib import Path

from PIL import Image

from config import MAX_RETRIES_BEFORE_RESTART, MAX_ROUNDS
from evaluator import evaluate
from fix_strategy import generate_fix_instruction
from generator import continue_conversation, start_conversation
from models import AgentResult, Evaluation, Requirements
from prompt_builder import build_initial_prompt

logger = logging.getLogger(__name__)


class LivestreamAgent:
    def __init__(self, output_dir: str = "output", max_rounds: int | None = None, max_retries_before_restart: int | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds if max_rounds is not None else MAX_ROUNDS
        self.max_retries_before_restart = max_retries_before_restart if max_retries_before_restart is not None else MAX_RETRIES_BEFORE_RESTART

    def run(self, reference_path: str, product_paths: list[str], requirements: Requirements) -> AgentResult:
        reference_bytes = Path(reference_path).read_bytes()
        product_bytes_list = [Path(p).read_bytes() for p in product_paths]

        with Image.open(reference_path) as ref_img:
            self.target_size = ref_img.size

        initial_prompt = build_initial_prompt(requirements, reference_path)
        logger.info(f"Starting initial generation... target size: {self.target_size}, products: {len(product_paths)}")
        logger.info(f"Config: max_rounds={self.max_rounds}, restart_after={self.max_retries_before_restart} failed fixes")

        image_paths = [reference_path] + product_paths
        conv = start_conversation(initial_prompt, image_paths)
        current_image = conv.latest_image
        self._save_intermediate(current_image, 0)
        logger.info("Round 0: initial image generated")

        best_image = current_image
        best_score = 0.0
        consecutive_failures = 0

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"Round {round_num}: evaluating...")
            evaluation = evaluate(current_image, reference_bytes, product_bytes_list, requirements)
            logger.info(
                f"Round {round_num}: scores={evaluation.scores.model_dump()}, "
                f"total={evaluation.total_score:.1f}, passed={evaluation.passed}"
            )

            if evaluation.issues:
                for issue in evaluation.issues:
                    logger.info(f"  Issue: {issue}")

            if evaluation.passed:
                final_path = self._save_final(current_image)
                logger.info(f"Passed at round {round_num}! Saved to {final_path}")
                return AgentResult(image_path=str(final_path), rounds=round_num, evaluation=evaluation)

            if evaluation.total_score > best_score:
                best_score = evaluation.total_score
                best_image = current_image

            consecutive_failures += 1

            if consecutive_failures >= self.max_retries_before_restart:
                logger.info(f"Round {round_num}: {consecutive_failures} consecutive failures, restarting fresh...")
                consecutive_failures = 0
                try:
                    conv = start_conversation(initial_prompt, image_paths)
                    current_image = conv.latest_image
                    self._save_intermediate(current_image, round_num)
                    logger.info(f"Round {round_num}: fresh restart generated")
                except Exception as e:
                    logger.warning(f"Round {round_num}: fresh restart failed: {e}")
                continue

            fix_instruction = generate_fix_instruction(evaluation)
            logger.info(f"Round {round_num}: applying fix...")

            try:
                conv = continue_conversation(conv, fix_instruction)
                current_image = conv.latest_image
                self._save_intermediate(current_image, round_num)
            except Exception as e:
                logger.warning(f"Round {round_num}: fix generation failed: {e}")
                continue

        final_path = self._save_final(best_image)
        logger.info(f"Max rounds reached. Best result saved to {final_path}")
        return AgentResult(image_path=str(final_path), rounds=self.max_rounds, partial=True)

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
