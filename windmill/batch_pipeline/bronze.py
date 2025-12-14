# bronze.py
import os
from u.youruser.utils import (
    fetch_csv_from_minio,
    get_postgres_conn,
    ensure_table_dynamic,
    insert_df,
)


def main():
    # Nome do bucket no MinIO onde o CSV está salvo
    bucket = os.getenv("MINIO_BUCKET", "aurora")

    # Prefixo opcional dentro do bucket (ex: pastas virtuais)
    prefix = os.getenv("MINIO_PREFIX", "")

    # Nome exato do arquivo CSV a ser carregado
    object_name = os.getenv(
        "MINIO_OBJECT", "mental_health_workplace_survey.csv"
    )

    # Lê o CSV diretamente do MinIO e retorna como DataFrame Polars
    df = fetch_csv_from_minio(
        bucket=bucket, prefix=prefix or None, object_name=object_name
    )

    # Abre conexão com Postgres
    conn = get_postgres_conn()
    cur = conn.cursor()

    # Cria a tabela Bronze com o schema dinâmico baseado no CSV original,
    # mantendo exatamente os nomes e tipos das colunas da fonte.
    ensure_table_dynamic(cur, "bronze_employees", df)

    # Insere os dados sem nenhum tipo de tratamento ou renomeação
    insert_df(conn, df, "bronze_employees")

    cur.close()
    conn.close()
    print(f"[bronze] inserted {df.height} rows into bronze_employees")


if __name__ == "__main__":
    main()
