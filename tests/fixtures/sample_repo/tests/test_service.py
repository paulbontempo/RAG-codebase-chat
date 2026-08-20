import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import UserService  # noqa: E402


def test_create_and_describe_user():
    service = UserService()
    service.create_user(1, "grace@example.com")
    assert service.describe_user(1) == "Entity#1 <grace@example.com>"


def test_get_user_missing_returns_none():
    service = UserService()
    assert service.get_user(999) is None
