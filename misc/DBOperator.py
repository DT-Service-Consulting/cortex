import os
from typing import Optional
from urllib.parse import quote_plus

import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class DBOperator:
    def __init__(
        self,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        engine: Optional[Engine] = None,
    ):
        if engine is not None:
            self._engine = engine
            return

        server = server or os.environ["AZURE_SQL_SERVER"]
        database = database or os.environ["AZURE_SQL_DATABASE"]
        username = username or os.environ["AZURE_SQL_USERNAME"]
        password = password or os.environ["AZURE_SQL_PASSWORD"]
        odbc = (
            f"Driver={{ODBC Driver 18 for SQL Server}};"
            f"Server=tcp:{server},1433;"
            f"Database={database};"
            f"Uid={username};"
            f"Pwd={password};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
        )
        self._engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}")

    def query(self, sql: str) -> pl.DataFrame:
        return pl.read_database(sql, connection=self._engine)

    def execute(self, sql: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(sql))
