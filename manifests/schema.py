from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime

class ManifestSchema(BaseModel):
    run_id: UUID
    git_commit_hash: str
    config_hash: str
    seed: int
    library_versions: dict[str, str]
    hardware: dict
    start_time: datetime
    end_time: datetime
    checkpoint_sha256: Optional[str] = None
