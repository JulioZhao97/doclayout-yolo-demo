"""Opt-in true-BF16 trainer: bfloat16 autocast with no gradient scaling.

The stock trainer runs ``autocast(self.amp)``, which defaults to float16, and
pairs it with a ``GradScaler``.  That pairing is what turned an attention
overflow into a delayed blow-up: every overflowing step was skipped and the
scale halved, until the scale reached its floor (~2**-22), at which point the
accumulated oversized gradient finally landed on the weights.

bfloat16 carries the exponent range of float32, so loss scaling is unnecessary
-- this is how bf16 training is normally done.  This trainer therefore:

* disables the stock float16 autocast and wraps the model's forward in
  ``torch.autocast("cuda", dtype=torch.bfloat16)`` instead, and
* swaps the ``GradScaler`` for a no-op shim, so the inherited guard telemetry
  keeps emitting the same JSONL fields with no live scaler behind them.

``Stage1GuardTrainer``, ``MineruSyncBNDetectionTrainer`` and every existing
attention module are left unchanged.
"""

from __future__ import annotations

import torch

from tools.stage1.stage1_guard_trainer import Stage1GuardTrainer


class _NoOpGradScaler:
    """Stand-in for ``torch.amp.GradScaler`` when training in bfloat16.

    Implements the methods the trainer and the guard telemetry call.
    ``get_scale()`` returns a constant 1.0, so the recorded
    ``amp_scale_before``/``amp_scale_after`` stay equal and no step is ever
    reported as skipped.  That is accurate rather than cosmetic: without loss
    scaling there is no scaler-driven skip mechanism left to report.
    """

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        return optimizer.step()

    def update(self, new_scale=None):
        return None

    def get_scale(self):
        return 1.0

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None

    def is_enabled(self):
        return False


class MineruBF16AutocastTrainer(Stage1GuardTrainer):
    """Sampled telemetry under true bfloat16 autocast. SyncBN deliberately OFF.

    SyncBN was measured to be the dominant cost of this experiment: with
    everything else held fixed, turning it on moved the step time from 0.780 to
    2.159 s/step (+177%), which is 96% of the total slowdown versus the original
    fp16 baseline.  Per-GPU batch is 160/8 = 20, so plain BatchNorm normalizes
    over 20 samples per rank instead of 160 globally -- accepted here in
    exchange for ~20 h/epoch instead of ~34.

    ``use_sync_bn`` is left unset, so the guarded hook in ``trainer.py``
    (``getattr(self, "use_sync_bn", False)``) skips the SyncBatchNorm
    conversion entirely.
    """

    def _setup_train(self, world_size):
        super()._setup_train(world_size)

        # The base setup has already built the fp16 scaler and wrapped the model
        # in DDP; adjust afterwards so the inherited construction path itself is
        # untouched.
        self.scaler = _NoOpGradScaler()

        # Turn off the stock autocast in _do_train.  Only the bfloat16 autocast
        # installed below then applies, so no float16 region remains.
        self.amp = False

        model = self.model
        if getattr(model, "_bf16_autocast_wrapped", False):
            return
        inner_forward = model.forward

        def forward(*args, **kwargs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return inner_forward(*args, **kwargs)

        model.forward = forward
        model._bf16_autocast_wrapped = True

    def optimizer_step(self):
        """Sample the guard telemetry instead of recording every step.

        ``Stage1GuardTrainer.optimizer_step`` gathers BN statistics and weight
        norms unconditionally, and each of those gathers copies a GPU result to
        the host -- a synchronization point that stalls the pipeline.
        ``STAGE1_GUARD_TRACE_INTERVAL`` was read by the base class but never
        actually used to skip anything; honour it here.

        On a non-sampled step the optimizer update is performed directly, in the
        same order as the stock trainer (unscale -> clip -> step -> zero grad ->
        EMA), with no host synchronization at all.  Sampled steps fall through
        to the parent and record the full record as before.
        """
        interval = max(1, int(getattr(self, "guard_trace_interval", 1)))
        if interval > 1 and (self.guard_step + 1) % interval != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.guard_step += 1
            return
        super().optimizer_step()


class MineruBF16AutocastSyncBNTrainer(MineruBF16AutocastTrainer):
    """The SyncBN variant, kept for reference and for A/B comparisons."""

    use_sync_bn = True


# Ultralytics' DDP launcher re-imports the trainer class by module name in every
# child process, so this must match the file's installed import path.
_MODULE = "ultralytics.models.yolo.detect.train_mineru_bf16_autocast_syncbn"
MineruBF16AutocastTrainer.__module__ = _MODULE
MineruBF16AutocastSyncBNTrainer.__module__ = _MODULE
