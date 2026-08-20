from utils import log_call


def notify(user_email: str) -> None:
    """Notify a user; imports UserService lazily to avoid a circular import at module load time."""
    from service import UserService  # noqa: PLC0415 (intentional: circular-import fixture case)

    log_call("notify")
    print(f"notifying {user_email} via {UserService.__name__}")
