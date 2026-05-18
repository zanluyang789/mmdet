"""
自定义 mmengine Hook
====================

CkptSyncHook
    每次 CheckpointHook 保存 .pth 后，把 work_dir 下的新增 .pth 文件实时同步
    到 ckpt_dir。

    为什么不直接用 CheckpointHook(out_dir=ckpt_dir)：
        mmengine 在 out_dir != work_dir 时会自动 append basename(work_dir) 子目录
        （为避免不同任务的 ckpt 撞在一起）。我们的场景下用户希望 ckpt 直接落在
        task.conf 的 checkpoint_path 下，所以单独写一个 sync hook 解决。
"""

from __future__ import annotations

import os
import shutil

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class CkptSyncHook(Hook):
    """
    监听 work_dir 下的 .pth / last_checkpoint，新增/更新就复制（或软链）到 ckpt_dir。

    Args:
        ckpt_dir: 目标目录（== task.conf 里的 checkpoint_path）
        link:     True 用 symlink；默认 False 走 copy（分布式存储更稳）
    """

    priority = "VERY_LOW"   # 在 CheckpointHook 之后

    def __init__(self, ckpt_dir: str, link: bool = False) -> None:
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.link = link
        # (filename, mtime) 已同步集合
        self._synced: set = set()

    # ---------- 内部 ----------
    def _sync_once(self, runner) -> None:
        try:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        except OSError as e:
            runner.logger.warning(f"[CkptSync] mkdir {self.ckpt_dir} failed: {e}")
            return
        try:
            entries = os.listdir(runner.work_dir)
        except OSError:
            return

        for name in entries:
            # 同步 *.pth 和 last_checkpoint 这种小文本指针
            if not (name.endswith(".pth") or name == "last_checkpoint"):
                continue
            src = os.path.join(runner.work_dir, name)
            if not os.path.isfile(src):
                continue
            try:
                mtime = os.path.getmtime(src)
            except OSError:
                continue
            key = (name, mtime)
            if key in self._synced:
                continue

            dst = os.path.join(self.ckpt_dir, name)
            try:
                if os.path.lexists(dst):
                    try:
                        if os.path.getmtime(dst) >= mtime:
                            self._synced.add(key)
                            continue
                    except OSError:
                        pass
                    os.remove(dst)
                if self.link:
                    os.symlink(src, dst)
                else:
                    shutil.copy2(src, dst)
                self._synced.add(key)
                runner.logger.info(f"[CkptSync] {name} -> {self.ckpt_dir}")
            except Exception as e:
                runner.logger.warning(f"[CkptSync] sync {name} failed: {e}")

    # ---------- Hook 接口（用宽松签名兼容 mmengine 版本差异） ----------
    def after_train_iter(self, runner, *args, **kwargs):
        self._sync_once(runner)

    def after_val_epoch(self, runner, *args, **kwargs):
        # best ckpt 通常在 val 之后保存
        self._sync_once(runner)

    def after_train(self, runner):
        self._sync_once(runner)

    def after_run(self, runner):
        self._sync_once(runner)
