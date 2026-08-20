from models import User
from notifier import notify
from utils import log_call, retry


class UserService:
    def __init__(self):
        self._users: dict[int, User] = {}

    @retry(times=2)
    def create_user(self, entity_id: int, email: str) -> User:
        log_call("create_user")
        user = User(entity_id, email)
        self._users[entity_id] = user
        notify(user.email)
        return user

    def get_user(self, entity_id: int) -> User | None:
        return self._users.get(entity_id)

    def describe_user(self, entity_id: int) -> str:
        user = self.get_user(entity_id)
        if user is None:
            raise KeyError(entity_id)
        return user.describe()
