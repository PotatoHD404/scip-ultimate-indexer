from pkg.models import User


class GreetingService:
    """Greets users and prepares API payloads."""

    def build_greeting(self, user: User, excited: bool = False) -> str:
        greeting = f"Hello, {user.name}"
        if excited:
            greeting = greeting + "!"
        return greeting


def serialize_user(user: User) -> dict[str, str]:
    return user.as_dict()
