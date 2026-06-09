# simple_emitter.py — Minimal EventEmitter for Python
# Matches Node.js EventEmitter API: .on(event, callback), .emit(event, data)
import threading
from collections import defaultdict


class EventEmitter:
    def __init__(self):
        self._listeners = defaultdict(list)
        self._lock = threading.Lock()

    def on(self, event: str, callback) -> None:
        with self._lock:
            self._listeners[event].append(callback)

    def off(self, event: str, callback) -> None:
        with self._lock:
            if callback in self._listeners[event]:
                self._listeners[event].remove(callback)

    def emit(self, event: str, *args, **kwargs) -> None:
        with self._lock:
            listeners = list(self._listeners.get(event, []))
        for cb in listeners:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                print(f"[Emitter] Error in {event} listener: {e}")
