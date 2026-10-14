"""Opt-in BF16-attention trainer with ordinary per-rank BatchNorm."""

from tools.stage1.stage1_guard_trainer import Stage1GuardTrainer


class MineruBF16DetectionTrainer(Stage1GuardTrainer):
    """New telemetry trainer that deliberately leaves SyncBN disabled."""

    # The base trainer converts BatchNorm to SyncBatchNorm only when this
    # opt-in flag is true.  Keeping it explicit makes the control condition
    # auditable in the DDP child process.
    use_sync_bn = False
