# utils.py
"""
Módulo utilitário central para operações com:
 - MinIO (leitura e listagem de arquivos)
 - PostgreSQL (conexão, criação de tabelas, inserção e leitura)
 - Polars (tipagem dinâmica -> SQL)

Este arquivo é usado por todas as pipelines Bronze/Silver/Gold e também
pelos steps de Pulsar + ML Predicts.

As funções aqui foram projetadas para serem simples, robustas e adequadas
para execução dentro do Windmill.
"""

import os
import io
from minio import Minio
import polars as pl
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


# ---------------------------------------------------------------------
# Configuração padrão de conexão com POSTGRES
# Obtida via variáveis de ambiente (recomendado para Windmill e Docker)
# ---------------------------------------------------------------------
POSTGRES_CONN = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "dbname": os.getenv("POSTGRES_DB", "analytics"),
    "user": os.getenv("POSTGRES_USER", "user"),
    "password": os.getenv("POSTGRES_PASSWORD", "password"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}


# ---------------------------------------------------------------------
# MINIO CLIENT
# Cria um cliente MinIO usando credenciais/endpoint do ambiente.
# ---------------------------------------------------------------------
def get_minio_client():
    """
    Retorna um cliente MinIO inicializado via ENV.
    MINIO_SECURE controla se usa HTTPS.
    """
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure = os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes")

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


# ---------------------------------------------------------------------
# Leitura de CSV diretamente do MinIO.
# Pode ler arquivo específico OU buscar o mais recente.
# ---------------------------------------------------------------------
def fetch_csv_from_minio(
    bucket: str, prefix: str | None = None, object_name: str | None = None
) -> pl.DataFrame:
    """
    Lê um CSV do MinIO.

    - Se object_name for passado, lê exatamente aquele arquivo.
    - Caso contrário, lista todos os CSV sob o prefix e seleciona o mais recente.
    """

    client = get_minio_client()

    # --- modo 1: busca arquivo específico ---
    if object_name:
        obj = client.get_object(bucket, object_name)
        data = obj.read()
        obj.close()
        obj.release_conn()
        return pl.read_csv(io.BytesIO(data))

    # --- modo 2: lista arquivos e pega o mais novo ---
    objects = client.list_objects(bucket, prefix=prefix or "", recursive=True)

    latest = None
    for o in objects:
        if not o.object_name.lower().endswith(".csv"):
            continue
        if latest is None or o.last_modified > latest.last_modified:
            latest = o

    if latest is None:
        raise FileNotFoundError(
            "Nenhum arquivo CSV encontrado no bucket/prefix informado."
        )

    obj = client.get_object(bucket, latest.object_name)
    data = obj.read()
    obj.close()
    obj.release_conn()
    return pl.read_csv(io.BytesIO(data))


# ---------------------------------------------------------------------
# Conexão com Postgres
# ---------------------------------------------------------------------
def get_postgres_conn():
    """Retorna conexão psycopg2 usando as variáveis POSTGRES_ do ambiente."""
    return psycopg2.connect(**POSTGRES_CONN)


# ---------------------------------------------------------------------
# Conversão de tipos Polars → SQL
# Usado para criação de tabelas dinâmicas.
# ---------------------------------------------------------------------
def polars_dtype_to_sql(dtype) -> str:
    """
    Converte dtype do Polars para tipo SQL (PostgreSQL).
    Mantém regra simples, suficiente para pipelines bronze/silver.
    """
    s = str(dtype).lower()

    if "int" in s:
        return "BIGINT"
    if "float" in s:
        return "DOUBLE PRECISION"
    if "bool" in s:
        return "BOOLEAN"
    if "date" in s and "time" not in s:
        return "DATE"
    if "datetime" in s or "time" in s:
        return "TIMESTAMP"

    return "TEXT"


# ---------------------------------------------------------------------
# Criação dinâmica de tabela com base no DF (usada no bronze)
# ---------------------------------------------------------------------
def ensure_table_dynamic(cur, table_name: str, df: pl.DataFrame):
    """
    Cria tabela automaticamente, usando os nomes e tipos do dataframe.
    Não define PK, apenas cria as colunas.
    """
    col_defs = []
    for col, dtype in zip(df.columns, df.dtypes):
        sql_type = polars_dtype_to_sql(dtype)
        col_defs.append(
            sql.SQL("{} {}").format(
                sql.Identifier(col),
                sql.SQL(sql_type),
            )
        )

    create_sql = sql.SQL(
        "CREATE TABLE IF NOT EXISTS {} ({})"
    ).format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(col_defs),
    )

    cur.execute(create_sql)


# ---------------------------------------------------------------------
# Criação de tabela com schema explícito (usada no gold ml_predicts)
# ---------------------------------------------------------------------
def ensure_table_fixed_schema(cur, table_name: str, schema_sql: str):
    """
    Cria tabela com schema fixo passado como string.
    Exemplo:
      schema_sql = "id INT PRIMARY KEY, nome TEXT"
    """
    cur.execute(
        sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(table_name),
            sql.SQL(schema_sql),
        )
    )


# ---------------------------------------------------------------------
# Inserção de DataFrame no PostgreSQL (usando execute_values para velocidade)
# ---------------------------------------------------------------------
def insert_df(conn, df: pl.DataFrame, table: str, conflict_cols: list | None = None):
    """
    Insere polars.DataFrame no Postgres usando execute_values (rápido).
    
    - conflict_cols: lista de colunas para ON CONFLICT DO NOTHING.
    """
    if df.is_empty():
        return

    cols = list(df.columns)
    values = [tuple(row) for row in df.rows()]

    with conn.cursor() as cur:
        # ON CONFLICT (col1, col2) DO NOTHING
        if conflict_cols:
            query_sql = (
                sql.SQL(
                    "INSERT INTO {} ({}) VALUES %s ON CONFLICT ({}) DO NOTHING"
                )
                .format(
                    sql.Identifier(table),
                    sql.SQL(", ").join(map(sql.Identifier, cols)),
                    sql.SQL(", ").join(map(sql.Identifier, conflict_cols)),
                )
                .as_string(conn)
            )
        else:
            query_sql = (
                sql.SQL("INSERT INTO {} ({}) VALUES %s")
                .format(
                    sql.Identifier(table),
                    sql.SQL(", ").join(map(sql.Identifier, cols)),
                )
                .as_string(conn)
            )

        execute_values(cur, query_sql, values)

    conn.commit()


# ---------------------------------------------------------------------
# Leitura completa de uma tabela Postgres → Polars DataFrame
# ---------------------------------------------------------------------
def read_table_df(conn, table: str) -> pl.DataFrame:
    """
    Lê tabela inteira do Postgres e retorna um Polars DataFrame.
    Usado para depuração ou validação.
    """
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(table)))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

    import pandas as pd
    df = pl.from_pandas(pd.DataFrame(rows, columns=cols))
    return df


# ---------------------------------------------------------------------
# Execução direta (caso rode standalone)
# ---------------------------------------------------------------------
def main():
    print("utils.py OK – módulo carregado.")
