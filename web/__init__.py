import json
from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    STARTED = "started"
    ROUND_IMAGE = "round_image"
    EVALUATING = "evaluating"
    EVALUATION_DONE = "evaluation_done"
    FIXING = "fixing"
    RESTARTING = "restarting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressEvent:
    event: EventType
    round: int = 0
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"event": self.event.value, "round": self.round, "data": self.data}, ensure_ascii=False)
