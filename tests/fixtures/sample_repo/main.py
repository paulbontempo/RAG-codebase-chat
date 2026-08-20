from service import UserService


def run() -> None:
    service = UserService()
    service.create_user(1, "grace@example.com")
    print(service.describe_user(1))


if __name__ == "__main__":
    run()
