# Database Schema

The sample project stores tenants and users in SQLite.

## Tables

- `tenants(id, slug, created_at)`
- `users(id, tenant_id, email, display_name)`

Authentication joins users to tenants by `tenant_id`.
