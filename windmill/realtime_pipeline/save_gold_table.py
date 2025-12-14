"""
Lê o último arquivo de previsões gerado pelo passo pulsar_predict.py
e salva na tabela GOLD ml_predicts do Postgres.

Configurações:
  PIPELINE_OUT_DIR (default ./data)
  ML_PREDICTS_TABLE (default ml_predicts)
"""
import os
from pathlib import Path
import pandas as pd
import polars as pl
from datetime import datetime

from u.youruser.utils import get_postgres_conn, ensure_table_fixed_schema, insert_df

OUT_DIR = Path(os.getenv("PIPELINE_OUT_DIR", "./data"))
TABLE_NAME = os.getenv("ML_PREDICTS_TABLE", "ml_predicts")

def latest_predictions_file():
    """
    Retorna o arquivo de predictions mais recente no diretório.
    """
    files = sorted(OUT_DIR.glob("predictions_*.csv"), reverse=True)
    return files[0] if files else None

def main():
    file = latest_predictions_file()
    if file is None:
        print("[save_ml_predicts] nenhum arquivo de predictions encontrado em", OUT_DIR)
        return

    df = pd.read_csv(file)
    if df.empty:
        print("[save_ml_predicts] arquivo vazio, abortando")
        return

    # Prepara o dataframe para inserir no banco
    # Mantém id_funcionario se existir, senão cria None
    insert_df_df = df.copy()
    if "id_funcionario" not in insert_df_df.columns:
        insert_df_df["id_funcionario"] = None

    # As "raw features" vão como JSONB no Postgres
    if "_raw_features" in insert_df_df.columns:
        insert_df_df["raw_features"] = insert_df_df["_raw_features"]
    else:
        insert_df_df["raw_features"] = insert_df_df.apply(lambda r: r.to_dict(), axis=1)

    if "predicted_at" not in insert_df_df.columns:
        insert_df_df["predicted_at"] = datetime.utcnow().isoformat()

    # Colunas que queremos carregar para o Postgres
    to_keep = ["id_funcionario", "prediction", "probability", "raw_features", "predicted_at"]
    insert_df_df = insert_df_df[[c for c in to_keep if c in insert_df_df.columns]]

    # Converte Pandas -> Polars (compatível com suas utils)
    try:
        pl_df = pl.from_pandas(insert_df_df)
    except Exception:
        pl_df = pl.DataFrame(insert_df_df)

    # Cria a tabela GOLD ml_predicts se não existir
    conn = get_postgres_conn()
    cur = conn.cursor()
    schema_sql = """
    id_pred SERIAL PRIMARY KEY,
    id_funcionario BIGINT,
    prediction INTEGER,
    probability DOUBLE PRECISION,
    raw_features JSONB,
    predicted_at TIMESTAMP
    """
    ensure_table_fixed_schema(cur, TABLE_NAME, schema_sql)
    conn.commit()
    cur.close()

    # Inserção simples sem upsert (predições são sempre novas)
    insert_df(conn, pl_df, TABLE_NAME, conflict_cols=[])
    conn.close()

    print(f"[save_ml_predicts] inserido {len(pl_df)} linhas em {TABLE_NAME}")

if __name__ == "__main__":
    main()
