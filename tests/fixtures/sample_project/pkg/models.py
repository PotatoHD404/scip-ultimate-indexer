class User:
    """Simple user domain object."""

    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "email": self.email}
