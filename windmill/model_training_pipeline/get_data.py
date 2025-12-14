# get_data.py
import os
from pathlib import Path
import polars as pl
from u.youruser.utils import get_postgres_conn, read_table_df

# Diretório onde o CSV será salvo. Permite override via variável de ambiente.
OUT_DIR = Path(os.getenv("PIPELINE_OUT_DIR", "./data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nome final do arquivo usado como input para a etapa de treinamento.
OUT_CSV = OUT_DIR / "silver_employees.csv"

def main():
    # Abre conexão com o Postgres para ler a tabela silver_employees (camada já tratada).
    conn = get_postgres_conn()
    df = read_table_df(conn, "silver_employees")
    conn.close()

    # Caso a tabela esteja vazia ou não exista, interrompe a etapa.
    if df is None or df.is_empty():
        print("[get_data] silver_employees is empty or not found.")
        return

    # Converte para Polars DataFrame (se ainda não for) e salva como CSV.
    try:
        df = pl.DataFrame(df) if not isinstance(df, pl.DataFrame) else df
        # Salva em CSV usando Polars (muito mais eficiente que pandas).
        df.write_csv(OUT_CSV)
        print(f"[get_data] saved {df.height} rows to {OUT_CSV}")
    except Exception as e:
        print("[get_data] failed to save CSV:", e)

if __name__ == "__main__":
    main()
