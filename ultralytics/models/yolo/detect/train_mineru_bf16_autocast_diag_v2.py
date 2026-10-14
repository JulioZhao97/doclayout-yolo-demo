#!/usr/bin/env python3
"""BF16-autocast trainer with non-finite step skipping + full event recording (v2).

Motivated by the v1 evidence (run ``..._diag_v1``): at step 4936 exactly one
rank (rank4) produced ``box_loss=+inf`` while its forward output, cls term and
BN stats were all clean. That inf crossed the DDP gradient allreduce, became
nan on every rank, and by 4937 the whole network was poisoned. **The bad step
was isolated** -- steps 4880/4900/4920 showed grad_norm 1858-2266 and
``var_max~10``, perfectly healthy, with no run-up.

So the divergence is not "training slowly goes bad"; it is a single overflowing
step that the training loop had no defence against:

* ``_NoOpGradScaler.step()`` is an unconditional ``optimizer.step()``.
* ``clip_grad_norm_`` defaults to ``error_if_nonfinite=False``, so a nan
  ``total_norm`` yields a nan ``clip_coef`` which is multiplied into *every*
  parameter.

This trainer closes that gap:

1. ``clip_grad_norm_`` result is checked with ``torch.isfinite``. If it is not
   finite the optimizer step is **skipped** and grads are zeroed, leaving
   weights and EMA untouched. Nothing else about the step changes.
2. Every skip is recorded -- both a dedicated ``skipped_steps_rank{rank}.jsonl``
   event and the parent's full guard record (forced through regardless of the
   sampling interval), so a skip can never happen silently.

Skipping is inherently rank-consistent: DDP averages gradients during backward,
so a non-finite gradient on one rank makes every rank's ``total_norm``
non-finite. All ranks reach the same decision without extra communication, and
no additional collective is introduced (avoiding any risk of a hang).

The point of the skip is *not* to paper over the bug -- it is to keep the run
alive past the bad step so ``DiagnosticDetectionLossV2`` can collect many
independent overflow samples instead of the single one a dead run yields.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from tools.stage1.loss_diagnostic_v2 import DiagnosticDetectionLossV2
from ultralytics.models.yolo.detect.train_mineru_bf16_autocast_syncbn import MineruBF16AutocastTrainer
from ultralytics.utils.torch_utils import de_parallel


def _finite_or_str(value):
    """Mirror ``stage1_guard_trainer._finite``: JSON cannot hold nan/inf."""
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return float(value)


class MineruBF16AutocastDiagV2Trainer(MineruBF16AutocastTrainer):
    """BF16-autocast trainer + non-finite step skip + fine-grained loss capture."""

    def _setup_train(self, world_size):
        super()._setup_train(world_size)

        rank = int(os.environ.get("RANK", 0))
        diag_dir = Path(self.save_dir) / "loss_diag"
        diag_dir.mkdir(parents=True, exist_ok=True)

        self.diag_rank = rank
        self.skip_events_path = diag_dir / f"skipped_steps_rank{rank}.jsonl"
        self.skipped_step_count = 0
        self.max_skip_events = int(os.environ.get("STAGE1_MAX_SKIP_EVENTS", "200"))

        max_events = int(os.environ.get("STAGE1_MAX_LOSS_EVENTS", "40"))
        raw_model = de_parallel(self.model)
        # ``DiagnosticDetectionLossV2.__init__`` performs the same setup as
        # ``v8DetectionLoss.__init__``, so assigning it directly replaces what
        # ``init_criterion()`` would have lazily built on the first loss call.
        self.diag_criterion = DiagnosticDetectionLossV2(
            raw_model, diag_dir=diag_dir, rank=rank, max_events=max_events
        )
        raw_model.criterion = self.diag_criterion

    def _record_skip(self, grad_norm, reason):
        """Persist every skipped step -- a silent skip would be worse than a crash."""
        self.skipped_step_count += 1
        if self.skipped_step_count > self.max_skip_events:
            return
        record = {
            "kind": "skipped_step",
            "time": time.time(),
            "rank": getattr(self, "diag_rank", 0),
            "step": self.guard_step + 1,
            "epoch_zero_based": int(self.epoch),
            "reason": reason,
            "grad_norm_before_clip": _finite_or_str(float(grad_norm)),
            "loss": _finite_or_str(float(self.loss.detach().float().item())),
            "loss_items": [_finite_or_str(float(x)) for x in self.loss_items.detach().float().cpu().reshape(-1)],
            "skipped_step_count": self.skipped_step_count,
            "loss_event_count": getattr(getattr(self, "diag_criterion", None), "event_count", None),
        }
        with self.skip_events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def optimizer_step(self):
        """Clip, then skip the update entirely if the gradient norm is not finite.

        Mirrors ``Stage1GuardTrainer.optimizer_step`` for the healthy path
        (same clip threshold, same EMA update, same guard record) and diverges
        only when ``total_norm`` is non-finite.
        """
        if getattr(self, "diag_criterion", None) is not None:
            self.diag_criterion.current_step = self.guard_step + 1

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

        if not bool(torch.isfinite(grad_norm).all().item()):
            # Drop this step: no optimizer.step(), no EMA update, no nan
            # clip_coef multiplied into the weights.  Weights and EMA are
            # exactly what they were before this batch.
            self.optimizer.zero_grad()
            self.guard_step += 1
            self._record_skip(grad_norm, "nonfinite_grad_norm")
            self._write_full_guard_record(grad_norm, skipped=True)
            return

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        self.guard_step += 1

        interval = max(1, int(getattr(self, "guard_trace_interval", 1)))
        if interval <= 1 or self.guard_step % interval == 0:
            self._write_full_guard_record(grad_norm, skipped=False)

    def _write_full_guard_record(self, grad_norm, skipped):
        """Reuse the parent guard record shape so existing trace tooling still works."""
        from tools.stage1.stage1_guard_trainer import _bn_stats

        record = {
            "kind": "step",
            "time": time.time(),
            "rank": getattr(self, "diag_rank", 0),
            "step": self.guard_step,
            "epoch_zero_based": int(self.epoch),
            "lr": [float(group["lr"]) for group in self.optimizer.param_groups],
            "loss": _finite_or_str(float(self.loss.detach().float().item())),
            "loss_items": [_finite_or_str(float(x)) for x in self.loss_items.detach().float().cpu().reshape(-1)],
            "grad_norm_before_clip": _finite_or_str(float(grad_norm)),
            "grad_clipped": bool(torch.isfinite(grad_norm).all().item() and float(grad_norm) > 10.0),
            "optimizer_step_skipped": bool(skipped),
            "skipped_step_count": getattr(self, "skipped_step_count", 0),
            "amp_scale_before": float(self.scaler.get_scale()),
            "amp_scale_after": float(self.scaler.get_scale()),
            "amp_step_skipped": False,
            "amp_skip_streak": 0,
            "selected_weight_rms": self._guard_selected_rms(),
            "bn_model": _bn_stats(self.model),
            "bn_ema": _bn_stats(self.ema.ema if self.ema is not None else None),
        }
        self._guard_write(record)


_MODULE = "ultralytics.models.yolo.detect.train_mineru_bf16_autocast_diag_v2"
MineruBF16AutocastDiagV2Trainer.__module__ = _MODULE
