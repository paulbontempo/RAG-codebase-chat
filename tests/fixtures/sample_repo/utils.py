import functools
import time


def retry(times: int = 3):
    """Decorator: retry a function call up to `times` on exception."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for _ in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    time.sleep(0)
            raise last_exc

        return wrapper

    return decorator


def log_call(name: str) -> None:
    print(f"called: {name}")
