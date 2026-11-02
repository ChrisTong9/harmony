from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np
import torch
@dataclass
class SAM3MaskResult:
    frames :Any
    mask: torch.Tensor


class SAM3MaskProvider:
    def __init__(self):
        self.video_predictor = None

    def _lazy_load(self) -> None:
        if self.video_predictor is not None:
            return

        from sam3.model_builder import build_sam3_video_predictor
        self.video_predictor = build_sam3_video_predictor()


    def generate_mask(self,video_path:str, visual_prompt:str) -> SAM3MaskResult:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if video_file.suffix.lower() not in [".mp4", ".webm", ".mov", ".mkv"]:
            raise ValueError(f"Unsupported video format: {video_file.suffix}")

        if not visual_prompt.strip():
            raise ValueError("Visual prompt cannot be empty for video mode.")

        self._lazy_load()


        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(video_file))
        frames = decoder[:]


        response = self.video_predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(video_file),
            }
        )

        session_id = response["session_id"]
        masks = []
        num_frames = len(frames)
        for frame_idx in range(num_frames):
            response = self.video_predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": frame_idx,
                    "text": visual_prompt,
                }
            )
            outputs = response["outputs"]
            out_binary_masks = outputs["out_binary_masks"]

            masks.append(self._normalize_mask(out_binary_masks))

        mask_np = np.concatenate(masks, axis=0)
        mask = torch.from_numpy(mask_np).unsqueeze(1)


        return SAM3MaskResult(
            frames=frames,
            mask=mask,
        )


    def _normalize_mask(self, out_binary_masks: Any) -> np.ndarray:
        """
        Try to normalize SAM3 output mask into shape [1, H, W].

        SAM3 output shape may differ slightly depending on version.
        This function keeps the first mask if multiple instances are returned.
        """

        if isinstance(out_binary_masks, torch.Tensor):
            mask = out_binary_masks.detach().cpu().numpy()
        else:
            mask = np.asarray(out_binary_masks)


        if mask.ndim == 4:

            mask = mask[0, 0]
        elif mask.ndim == 3:

            mask = mask[0]
        elif mask.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected SAM3 mask shape: {mask.shape}")

        mask = mask.astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)


        return mask[None, :, :]
