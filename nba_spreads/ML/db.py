"""
Database helpers for loading NBA tables used by modeling code.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine


def get_engine_from_env() -> "object":
    """
    Create a SQLAlchemy engine from the ``DATABASE_URL`` environment variable.

    :raises ValueError: If ``DATABASE_URL`` is missing.
    :return: A SQLAlchemy Engine (typed as object to avoid importing sqlalchemy.engine).
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL not found in environment. Set it in your shell or .env file."
        )
    return create_engine(database_url)


def load_game_logs(engine: "object") -> pd.DataFrame:
    """
    Load the raw `nba.game_logs` table into a pandas DataFrame.

    :param engine: SQLAlchemy engine connected to the database.
    :return: DataFrame containing all columns from `nba.game_logs`.
    """
    query = "SELECT * FROM nba.game_logs"
    df = pd.read_sql(query, con=engine)
    return df

