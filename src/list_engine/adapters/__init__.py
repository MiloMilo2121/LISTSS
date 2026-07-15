"""Infrastructure adapters selected at the application composition boundary."""

from list_engine.adapters.postgres import PostgresCompanyRepository

__all__ = ["PostgresCompanyRepository"]
