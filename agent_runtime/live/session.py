from dataclasses import dataclass, field
import time
import uuid


@dataclass
class LiveSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    source: str = "rnos-runtime"
    mode: str = "live"

