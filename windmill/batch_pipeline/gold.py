# gold.py
import polars as pl
from u.youruser.utils import (
    get_postgres_conn,
    read_table_df,
    ensure_table_fixed_schema,
    insert_df,
)


def main():
    conn = get_postgres_conn()

    # Lê dados consolidados da Silver
    df = read_table_df(conn, "silver_employees")

    if df.is_empty():
        print("[gold] silver_employees is empty — nothing to do.")
        conn.close()
        return

    # Conjunto reduzido de colunas que realmente serão usadas na Gold,
    # representando uma versão mais analítica e objetiva da Silver.
    keep_cols = [
        "id_funcionario",
        "idade",
        "genero",
        "pais",
        "cargo",
        "departamento",
        "anos_na_empresa",
        "horas_trabalhadas_semana",
        "trabalho_remoto",
        "tempo_deslocamento",
        "possui_apoio_saude_mental",
        "faixa_salarial",
        "tamanho_equipe",
        "risco_burnout",
    ]

    # Seleciona apenas colunas que existem na tabela Silver
    df_gold = df.select([c for c in keep_cols if c in df.columns])

    # Casts para garantir tipos consistentes entre execuções
    casts = {
        "id_funcionario": pl.Int64,
        "idade": pl.Int64,
        "anos_na_empresa": pl.Int64,
        "horas_trabalhadas_semana": pl.Int64,
        "tempo_deslocamento": pl.Float64,
        "tamanho_equipe": pl.Int64,
        "risco_burnout": pl.Float64,
        "possui_apoio_saude_mental": pl.Utf8,
        "faixa_salarial": pl.Utf8,
        "trabalho_remoto": pl.Utf8,
        "genero": pl.Utf8,
        "pais": pl.Utf8,
        "cargo": pl.Utf8,
        "departamento": pl.Utf8,
    }

    # Aplica casts quando possível
    for col, dtype in casts.items():
        if col in df_gold.columns:
            try:
                df_gold = df_gold.with_columns(pl.col(col).cast(dtype))
            except Exception:
                pass  # mantém tipo atual se falhar

    # Criação da tabela Gold com schema fixo e limpo
    schema_sql = """
        id_funcionario BIGINT PRIMARY KEY,
        idade BIGINT,
        genero TEXT,
        pais TEXT,
        cargo TEXT,
        departamento TEXT,
        anos_na_empresa BIGINT,
        horas_trabalhadas_semana BIGINT,
        trabalho_remoto TEXT,
        tempo_deslocamento DOUBLE PRECISION,
        possui_apoio_saude_mental TEXT,
        faixa_salarial TEXT,
        tamanho_equipe BIGINT,
        risco_burnout DOUBLE PRECISION
    """

    cur = conn.cursor()
    ensure_table_fixed_schema(cur, "gold_employees", schema_sql)

    # Insere (upsert) o conjunto da Gold
    insert_df(conn, df_gold, "gold_employees", conflict_cols=["id_funcionario"])

    cur.close()
    conn.close()
    print(f"[gold] wrote {df_gold.height} rows into gold_employees")


if __name__ == "__main__":
    main()
