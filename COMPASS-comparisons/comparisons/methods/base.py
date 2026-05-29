from abc import ABC, abstractmethod
from pathlib import Path
import json
import logging
from typing import Any


class Method(ABC):
    name: str
    log: logging.Logger
    out_path: Path

    @abstractmethod
    def init_worker(
        self,
        log,
        scenario,
        out_path,
        clear_existing: bool = False,
        raise_on_err: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_rep(self, rep_id, **args) -> Any: ...

    def _load_done_reps(self) -> set[int]:
        idx = self._index_path()
        if not idx.exists():
            return set()
        done = set()
        with open(idx) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("ok"):
                    done.add(rec["rep_id"])
        return done

    def _index_path(self) -> Path:
        return self.out_path.with_suffix(".index.jsonl")


    def _append(self, record: dict, rep_id: int, ok: bool) -> None:
        # heavy record → main file
        with open(self.out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        # tiny index entry → sidecar
        with open(self._index_path(), "a") as f:
            f.write(json.dumps({"rep_id": rep_id, "ok": ok}) + "\n")
            f.flush()