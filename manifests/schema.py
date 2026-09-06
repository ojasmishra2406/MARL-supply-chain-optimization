from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ManifestSchema(BaseModel):
    run_id: UUID
    git_commit_hash: str
    config_hash: str
    seed: int
    library_versions: dict[str, str]
    hardware: dict
    start_time: datetime
    end_time: datetime
    checkpoint_sha256: str | None = None
