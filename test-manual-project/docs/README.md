# Project Documentation

## Overview

This project provides authentication and database services for web applications.

## Features

- User authentication with token-based sessions
- Database connection pooling
- Secure password hashing

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### Database Settings

Set the following environment variables:

- `DATABASE_PATH`: Path to SQLite database file
- `DB_POOL_SIZE`: Number of connections in pool (default: 5)

### Authentication

- `AUTH_SECRET_KEY`: Secret key for token generation
- `TOKEN_EXPIRY_HOURS`: Token validity period

## Usage

### Authentication Service

```python
from src.auth import create_auth_service

service = create_auth_service("your-secret-key")
token = service.authenticate("username", "password")
```

### Database Pool

```python
from src.database import create_database_pool

pool = create_database_pool("app.db")
results = pool.execute_query("SELECT * FROM users")
```

## API Reference

See [API Documentation](./api.md) for detailed API specifications.

## Contributing

Please read [CONTRIBUTING.md](./contributing.md) before submitting pull requests.
