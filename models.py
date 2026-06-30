from pydantic import BaseModel


class Requirements(BaseModel):
    description: str = ""


class EvaluationScores(BaseModel):
    pose_realism: int = 0
    product_accuracy: int = 0
    product_size: int = 0
    requirements_match: int = 0
    physics: int = 0


class Evaluation(BaseModel):
    scores: EvaluationScores = EvaluationScores()
    issues: list[str] = []
    passed: bool = False

    @property
    def total_score(self) -> float:
        s = self.scores
        return (s.pose_realism + s.product_accuracy + s.product_size + s.requirements_match + s.physics) / 5.0


class AgentResult(BaseModel):
    image_path: str
    rounds: int
    evaluation: Evaluation | None = None
    partial: bool = False
