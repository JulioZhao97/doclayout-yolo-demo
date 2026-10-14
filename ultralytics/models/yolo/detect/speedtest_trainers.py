"""Speed-test trainer variants for isolating the 1.83x slowdown.

Measurement-only.  Each variant changes exactly one thing relative to the
0.776 s/step baseline (original HWC2f + stock DetectionTrainer + fp16), so the
cost of that one change can be read directly off the step time.

* ``GuardFP16Trainer``  -- baseline model and precision, guard telemetry ON.
  Isolates the full cost of ``Stage1GuardTrainer``: the per-step BN sweep, the
  ``.item()`` synchronizations, and the per-rank JSONL writes to shared storage.
  Note SyncBN stays OFF (``Stage1GuardTrainer`` does not set ``use_sync_bn``),
  so this is guard overhead alone.

* ``BF16StockTrainer`` -- bfloat16 autocast with no loss scaling, guard
  telemetry OFF.  Isolates the cost of running the network in bfloat16.

Neither class is used by a real training run; both exist so the two hypotheses
can be priced separately instead of guessed at.
"""

from __future__ import annotations

import torch

from tools.stage1.stage1_guard_trainer import Stage1GuardTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.train_mineru_bf16_autocast_syncbn import _NoOpGradScaler


class _SampledGuardMixin:
    """Honour ``STAGE1_GUARD_TRACE_INTERVAL`` the way the real v2 run does.

    ``Stage1GuardTrainer`` reads the variable but never skips on it, so without
    this every variant would pay the full per-step telemetry cost and would not
    be comparable to the v2 measurement.
    """

    def optimizer_step(self):
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


class GuardFP16Trainer(_SampledGuardMixin, Stage1GuardTrainer):
    """#6: guard telemetry (sampled) + fp16, SyncBN OFF.

    Against #5 this isolates SyncBN; against the 0.777 baseline it prices the
    guard telemetry on the original block -- the measurement the earlier
    ``guard_fp16`` attempt left ambiguous.
    """


class NewBlockFP16SyncBNTrainer(_SampledGuardMixin, Stage1GuardTrainer):
    """#3/#4/#5: guard (sampled) + SyncBN + fp16.

    Used with three different model YAMLs so that the block is the only thing
    that changes between those rows.  The stock fp16 GradScaler is kept because
    that is what the original fp16 path uses; numerical divergence is expected
    and irrelevant, as these runs stop after a few hundred steps.
    """

    use_sync_bn = True


class NewBlockStockFP16Trainer(DetectionTrainer):
    """#7: the new block with NO guard and NO SyncBN, plain fp16 AMP.

    Differs from the 0.777 baseline only by the attention block, so this is the
    direct price of the block itself.
    """


class BF16StockTrainer(DetectionTrainer):
    """bfloat16 autocast, no loss scaling, no telemetry. SyncBN off."""

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        self.scaler = _NoOpGradScaler()
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


# The DDP launcher re-imports these by module name in every child process.
GuardFP16Trainer.__module__ = "ultralytics.models.yolo.detect.speedtest_trainers"
BF16StockTrainer.__module__ = "ultralytics.models.yolo.detect.speedtest_trainers"
NewBlockFP16SyncBNTrainer.__module__ = "ultralytics.models.yolo.detect.speedtest_trainers"
NewBlockStockFP16Trainer.__module__ = "ultralytics.models.yolo.detect.speedtest_trainers"
