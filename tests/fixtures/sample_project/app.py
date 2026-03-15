from pkg.models import User
from pkg.services import GreetingService, serialize_user


def build_demo_user(name: str) -> User:
    """Create a user for greeting demos."""
    return User(name=name, email=f"{name.lower()}@example.com")


def run() -> dict[str, str]:
    service = GreetingService()
    user = build_demo_user("Ada")
    service.build_greeting(user, excited=True)
    return serialize_user(user)


if __name__ == "__main__":
    print(run())
