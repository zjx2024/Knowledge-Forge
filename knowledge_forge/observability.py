"""Low-overhead, fail-open metrics for ingestion and graph construction.

The collector deliberately stores counts and durations only.  Source text,
prompts, credentials, and model responses never enter the metric payload.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import inspect
import logging
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Callable


LOGGER = logging.getLogger(__name__)


_CURRENT: contextvars.ContextVar["RunMetrics | None"] = contextvars.ContextVar(
    "knowledge_forge_run_metrics", default=None
)


@dataclasses.dataclass
class RunMetrics:
    kind: str
    counters: dict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    durations_ms: dict[str, float] = dataclasses.field(default_factory=lambda: defaultdict(float))
    gauges: dict[str, float] = dataclasses.field(default_factory=dict)
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[str(name)] += int(value)

    def duration(self, name: str, milliseconds: float) -> None:
        with self._lock:
            self.durations_ms[str(name)] += max(0.0, float(milliseconds))

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[str(name)] = float(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "kind": self.kind,
                "counters": dict(sorted(self.counters.items())),
                "durations_ms": {
                    key: round(value, 3) for key, value in sorted(self.durations_ms.items())
                },
                "gauges": dict(sorted(self.gauges.items())),
            }


def current_metrics() -> RunMetrics | None:
    return _CURRENT.get()


def increment(name: str, value: int = 1) -> None:
    recorder = current_metrics()
    if recorder is not None:
        recorder.increment(name, value)


def gauge(name: str, value: float) -> None:
    recorder = current_metrics()
    if recorder is not None:
        recorder.gauge(name, value)


@contextlib.contextmanager
def timed(name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        recorder = current_metrics()
        if recorder is not None:
            recorder.duration(name, (time.perf_counter() - started) * 1000)


@contextlib.contextmanager
def metrics_scope(kind: str) -> Iterator[RunMetrics]:
    existing = current_metrics()
    if existing is not None:
        yield existing
        return
    recorder = RunMetrics(kind=kind)
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT.reset(token)


class InstrumentedCursor:
    def __init__(self, cursor: Any):
        self._cursor = cursor

    def execute(self, statement: Any, params: Any = None, *args: Any, **kwargs: Any) -> Any:
        increment("db_statement_count")
        started = time.perf_counter()
        try:
            result = self._cursor.execute(statement, params, *args, **kwargs)
            return self if result is self._cursor else result
        finally:
            recorder = current_metrics()
            if recorder is not None:
                recorder.duration("db_statement", (time.perf_counter() - started) * 1000)

    def executemany(self, statement: Any, params_seq: Any, *args: Any, **kwargs: Any) -> Any:
        increment("db_statement_count")
        increment("db_executemany_count")
        started = time.perf_counter()
        try:
            result = self._cursor.executemany(statement, params_seq, *args, **kwargs)
            return self if result is self._cursor else result
        finally:
            recorder = current_metrics()
            if recorder is not None:
                recorder.duration("db_statement", (time.perf_counter() - started) * 1000)

    def __enter__(self) -> "InstrumentedCursor":
        self._cursor.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._cursor.__exit__(*args)

    def __iter__(self):
        return iter(self._cursor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class InstrumentedConnection:
    """Transparent psycopg proxy used only while a metric scope is active."""

    def __init__(self, connection: Any):
        self._connection = connection

    def execute(self, statement: Any, params: Any = None, *args: Any, **kwargs: Any) -> Any:
        increment("db_statement_count")
        started = time.perf_counter()
        try:
            return self._connection.execute(statement, params, *args, **kwargs)
        finally:
            recorder = current_metrics()
            if recorder is not None:
                recorder.duration("db_statement", (time.perf_counter() - started) * 1000)

    def cursor(self, *args: Any, **kwargs: Any) -> InstrumentedCursor:
        return InstrumentedCursor(self._connection.cursor(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def measured(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            increment(f"{name}_calls")
            with timed(name):
                return function(*args, **kwargs)

        wrapper.__name__ = function.__name__
        wrapper.__doc__ = function.__doc__
        wrapper.__module__ = function.__module__
        return wrapper

    return decorate


def recorded_job(kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Measure a service job and persist its aggregate without affecting it."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind_partial(*args, **kwargs)
            service = bound.arguments.get("self")
            document_id = bound.arguments.get("document_id")
            project_id = bound.arguments.get("project_id")
            job_id = bound.arguments.get("job_id")
            run_id = f"{kind}:{job_id or uuid.uuid4().hex}"
            def persist(snapshot: dict[str, Any]) -> None:
                database = getattr(service, "db", None)
                if project_id is None and document_id is not None and database is not None:
                    document = database.get_document_including_deleted(int(document_id))
                    resolved_project_id = (document or {}).get("project_id")
                else:
                    resolved_project_id = project_id
                if database is not None and hasattr(database, "record_operation_metrics"):
                    database.record_operation_metrics(
                        run_id=run_id,
                        kind=kind,
                        project_id=(
                            int(resolved_project_id) if resolved_project_id is not None else None
                        ),
                        document_id=int(document_id) if document_id is not None else None,
                        job_id=str(job_id) if job_id is not None else None,
                        metrics=snapshot,
                    )

            with metrics_scope(kind) as recorder:
                try:
                    result = function(*args, **kwargs)
                except BaseException:
                    snapshot = recorder.snapshot()
                    try:
                        persist(snapshot)
                    except Exception as exc:  # metrics must never change the business outcome
                        LOGGER.warning("Operation metrics persistence failed for %s: %s", run_id, exc)
                    raise
            try:
                persist(recorder.snapshot())
            except Exception as exc:  # metrics must never change the business outcome
                LOGGER.warning("Operation metrics persistence failed for %s: %s", run_id, exc)
            return result

        return wrapper

    return decorate
