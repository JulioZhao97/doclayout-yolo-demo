#!/usr/bin/env python3
"""BF16-autocast trainer + exact-step nan capture, for locating the step-4936 nan.

Two additions on top of ``MineruBF16AutocastTrainer``, both instrumentation
only -- no change to the optimizer/model/data path when nothing goes wrong:

1. ``model.criterion`` is swapped for ``DiagnosticDetectionLoss`` (see
   ``loss_diagnostic.py``), which writes a one-shot diagnostic JSON the first
   time any of {raw model output, assigner output, cls loss term, box loss
   term} turns non-finite -- telling us which stage broke first and what
   dtype the tensors were at that point.

2. ``optimizer_step`` keeps the parent's sampling behaviour (full guard
   record only every ``guard_trace_interval`` steps, to avoid syncing every
   step) but adds one cheap, unconditional ``torch.isfinite(self.loss)``
   check. The instant that check fails, the *current* step is forced through
   the full parent guard record regardless of the sampling interval, so we
   get complete BN/grad/weight telemetry for the exact step the divergence
   happens on instead of the next sampled point up to 19 steps later.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from tools.stage1.loss_diagnostic import DiagnosticDetectionLoss
from ultralytics.models.yolo.detect.train_mineru_bf16_autocast_syncbn import MineruBF16AutocastTrainer


class MineruBF16AutocastDiagTrainer(MineruBF16AutocastTrainer):
    """``MineruBF16AutocastTrainer`` plus exact-step nan capture."""

    def _setup_train(self, world_size):
        super()._setup_train(world_size)

        rank = int(os.environ.get("RANK", 0))
        diag_path = Path(self.save_dir) / f"guard_nan_diag_rank{rank}.json"

        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        # ``DiagnosticDetectionLoss.__init__`` runs the same setup as
        # ``v8DetectionLoss.__init__`` (builds its own assigner/bbox_loss), so
        # this replaces whatever ``init_criterion()`` would have lazily built
        # on the first ``model.loss()`` call -- no need to trigger that first.
        raw_model.criterion = DiagnosticDetectionLoss(raw_model, diag_path=diag_path, rank=rank)

    def optimizer_step(self):
        """Force the full guard record through on the exact step loss first goes non-finite."""
        interval = max(1, int(getattr(self, "guard_trace_interval", 1)))
        is_sample_step = interval <= 1 or (self.guard_step + 1) % interval == 0
        if not is_sample_step:
            loss_finite = bool(torch.isfinite(self.loss.detach()).all().item())
            if not loss_finite:
                super(MineruBF16AutocastTrainer, self).optimizer_step()  # Stage1GuardTrainer's full record
                return
        super().optimizer_step()


_MODULE = "ultralytics.models.yolo.detect.train_mineru_bf16_autocast_diag"
MineruBF16AutocastDiagTrainer.__module__ = _MODULE
