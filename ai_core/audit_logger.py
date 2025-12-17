# ai_core/audit_logger.py

import json
from pathlib import Path
from datetime import datetime, timezone


class AuditLogger:
    def __init__(self, path: str | Path = "logs/audit.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict) -> None:
        event["ts_utc"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
