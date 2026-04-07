# API Documentation

## Authentication Service

### Class: AuthenticationService

Main class for handling authentication operations.

#### Constructor

```python
AuthenticationService(secret_key: str)
```

**Parameters:**
- `secret_key` (str): Secret key used for token generation

#### Methods

##### authenticate

```python
authenticate(username: str, password: str) -> Optional[str]
```

Authenticate a user and return an authentication token.

**Returns:**
- `str` or `None`: Token if authentication successful, None otherwise

##### validate_session

```python
validate_session(session_id: str) -> bool
```

Check if a session is still valid.

**Returns:**
- `bool`: True if session is valid, False otherwise

##### logout

```python
logout(session_id: str) -> bool
```

Invalidate a session.

**Returns:**
- `bool`: True if session was found and invalidated

## Database Pool

### Class: DatabasePool

Connection pool for managing database connections.

#### Constructor

```python
DatabasePool(db_path: str, pool_size: int = 5)
```

**Parameters:**
- `db_path` (str): Path to SQLite database file
- `pool_size` (int): Number of connections in pool

#### Methods

##### execute_query

```python
execute_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]
```

Execute a SQL query and return results.

**Returns:**
- `List[Dict]`: List of dictionaries representing rows
