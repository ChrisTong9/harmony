from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioEditContext:
    user_prompt: str
    input_audio: Optional[str] = None
    output_audio: Optional[str] = None
    input_video: Optional[str] = None
    output_video: Optional[str] = None
    selected_script: Optional[str] = None
    params: dict = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    fallback_used: bool = False
    execution_error: Optional[str] = None
    execution_status: str = 'pending'
