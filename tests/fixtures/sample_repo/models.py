class BaseEntity:
    def __init__(self, entity_id: int):
        self.entity_id = entity_id

    def describe(self) -> str:
        return f"Entity#{self.entity_id}"


class User(BaseEntity):
    def __init__(self, entity_id: int, email: str):
        super().__init__(entity_id)
        self.email = email

    def describe(self) -> str:
        base = super().describe()
        return f"{base} <{self.email}>"
