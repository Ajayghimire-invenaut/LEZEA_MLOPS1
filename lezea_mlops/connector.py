# lezea_mlops/connector.py
from queue import Queue, Empty
from threading import Thread, Event
from contextlib import contextmanager
from typing import Optional, Dict, Any
from .tracker import ExperimentTracker

class LeZeAConnector:
    """
    Non-blocking wrapper around ExperimentTracker for training loops.
    Use async_mode=True to queue writes off the hot path.
    """
    def __init__(self, experiment_name: str, *, purpose: str = "", tags: Dict[str,str]|None=None,
                 async_mode: bool = True, strict: bool = False, prom_port: Optional[int] = None):
        self.tracker = ExperimentTracker(experiment_name, purpose=purpose, tags=tags or {})
        self.async_mode = async_mode
        self.strict = strict
        self._q: Queue[tuple[str, tuple, dict]] = Queue()
        self._stop = Event()
        self._worker: Optional[Thread] = None
        self.prom_port = prom_port

    def start(self):
        self.tracker.start(strict=self.strict, prom_port=self.prom_port)
        if self.async_mode:
            self._worker = Thread(target=self._drain, daemon=True)
            self._worker.start()
        return self

    def end(self):
        if self.async_mode:
            self._stop.set()
            if self._worker:
                self._worker.join(timeout=5)
        self.tracker.end()

    def _call(self, method: str, *a, **kw):
        if not self.async_mode:
            return getattr(self.tracker, method)(*a, **kw)
        self._q.put((method, a, kw))

    def _drain(self):
        while not self._stop.is_set() or not self._q.empty():
            try:
                m, a, kw = self._q.get(timeout=0.2)
                try:
                    getattr(self.tracker, m)(*a, **kw)
                except Exception as e:
                    # last-resort: write to local fallback?
                    self.tracker.logger.warning(f"connector dropped call {m}: {e}")
            except Empty:
                continue

    # ------- public API used by AGI code -------
    def log_step(self, step: int, metrics: Dict[str, float], *, sample_ids=None, split=None):
        self._call("log_training_step", step=step, metrics=metrics,
                   sample_ids=sample_ids, split=split)

    def log_checkpoint(self, path: str, *, step: Optional[int]=None, role="model", metadata=None):
        self._call("log_checkpoint", path, step=step, role=role, metadata=metadata)

    def log_results(self, *, tasker=None, builder=None, actions_outputs=None, step=None):
        self._call("log_results", tasker_rewards=tasker, builder_rewards=builder,
                   actions_outputs=actions_outputs, step=step)

    def log_mod_tree(self, tree: dict, *, scope: Optional[dict]=None, stats_only=False):
        self._call("log_modification_tree", tree=tree, scope=scope, stats_only=stats_only)

    def log_data_splits(self, train: int, val: int, test: int, extra: dict|None=None):
        self._call("log_data_splits", train, val, test, extra=extra)

    def log_lezea_config(self, **kw): self._call("log_lezea_config", **kw)
    def log_constraints(self, **kw):  self._call("log_constraints", **kw)

    @contextmanager
    def scope(self, level: str, entity_id: str):
        with self.tracker.scope(level, entity_id):
            yield
