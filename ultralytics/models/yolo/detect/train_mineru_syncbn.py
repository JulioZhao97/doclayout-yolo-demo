"""Opt-in SyncBatchNorm + per-step telemetry trainer for MinerU Stage1."""

from tools.stage1.stage1_guard_trainer import Stage1GuardTrainer


class MineruSyncBNDetectionTrainer(Stage1GuardTrainer):
    """New opt-in trainer with SyncBN and complete step-level telemetry.

    The base Ultralytics trainer and the original MinerU trainer remain
    unchanged.  SyncBN is enabled only for this class through the guarded
    DDP hook, while Stage1GuardTrainer writes loss/lr/gradient/weight/BN
    statistics for every optimizer step.
    """

    use_sync_bn = True
