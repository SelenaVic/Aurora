# silver.py
import polars as pl
from u.youruser.utils import (
    get_postgres_conn,
    read_table_df,
    ensure_table_fixed_schema,
    insert_df,
)


def main():
    conn = get_postgres_conn()

    # Lê toda a tabela Bronze para gerar a Silver
    df = read_table_df(conn, "bronze_employees")

    # Verifica se há dados a processar
    if df.is_empty():
        print("[silver] bronze_employees is empty — nothing to do.")
        conn.close()
        return

    # Mapeamento completo de tradução entre as colunas BRONZE e SILVER.
    # Apenas colunas que existirem serão traduzidas.
    translation_map = {
        "EmployeeID": "id_funcionario",
        "Age": "idade",
        "Gender": "genero",
        "Country": "pais",
        "JobRole": "cargo",
        "Department": "departamento",
        "YearsAtCompany": "anos_na_empresa",
        "WorkHoursPerWeek": "horas_trabalhadas_semana",
        "RemoteWork": "trabalho_remoto",
        "BurnoutLevel": "nivel_estresse_burnout",
        "JobSatisfaction": "satisfacao_trabalho",
        "StressLevel": "nivel_estresse",
        "ProductivityScore": "pontuacao_produtividade",
        "SleepHours": "horas_sono",
        "PhysicalActivityHrs": "horas_atividade_fisica",
        "CommuteTime": "tempo_deslocamento",
        "HasMentalHealthSupport": "possui_apoio_saude_mental",
        "ManagerSupportScore": "pontuacao_suporte_gestor",
        "HasTherapyAccess": "possui_acesso_terapia",
        "MentalHealthDaysOff": "dias_afastamento_saude_mental",
        "SalaryRange": "faixa_salarial",
        "WorkLifeBalanceScore": "pontuacao_equilibrio_vida_trabalho",
        "TeamSize": "tamanho_equipe",
        "CareerGrowthScore": "pontuacao_crescimento_carreira",
        "BurnoutRisk": "risco_burnout",
    }

    # Filtra apenas as colunas que realmente existem no Bronze
    rename_dict = {k: v for k, v in translation_map.items() if k in df.columns}

    # Renomeia para o padrão da Silver
    df_silver = df.rename(rename_dict)

    # Conversões de tipo para padronização da Silver
    casts = {
        "id_funcionario": pl.Int64,
        "idade": pl.Int64,
        "anos_na_empresa": pl.Int64,
        "horas_trabalhadas_semana": pl.Int64,
        "nivel_estresse_burnout": pl.Float64,
        "satisfacao_trabalho": pl.Float64,
        "nivel_estresse": pl.Float64,
        "pontuacao_produtividade": pl.Float64,
        "horas_sono": pl.Float64,
        "horas_atividade_fisica": pl.Float64,
        "tempo_deslocamento": pl.Float64,
        "pontuacao_suporte_gestor": pl.Float64,
        "dias_afastamento_saude_mental": pl.Int64,
        "pontuacao_equilibrio_vida_trabalho": pl.Float64,
        "tamanho_equipe": pl.Int64,
        "pontuacao_crescimento_carreira": pl.Float64,
        "risco_burnout": pl.Int64,
    }

    # Aplica casts quando possível (evita erros silenciosos)
    for col, dtype in casts.items():
        if col in df_silver.columns:
            try:
                df_silver = df_silver.with_columns(pl.col(col).cast(dtype))
            except Exception:
                pass  # Se falhar, mantém como está

    # Mantém apenas as colunas esperadas da Silver
    expected_cols = list(translation_map.values())
    df_silver = df_silver.select([c for c in expected_cols if c in df_silver.columns])

    # Cria tabela Silver com schema fixo (garante consistência ao longo do tempo)
    cur = conn.cursor()
    ensure_table_fixed_schema(
        cur,
        "silver_employees",
        """
        id_funcionario BIGINT PRIMARY KEY,
        idade BIGINT,
        genero TEXT,
        pais TEXT,
        cargo TEXT,
        departamento TEXT,
        anos_na_empresa BIGINT,
        horas_trabalhadas_semana BIGINT,
        trabalho_remoto TEXT,
        nivel_estresse_burnout DOUBLE PRECISION,
        satisfacao_trabalho DOUBLE PRECISION,
        nivel_estresse DOUBLE PRECISION,
        pontuacao_produtividade DOUBLE PRECISION,
        horas_sono DOUBLE PRECISION,
        horas_atividade_fisica DOUBLE PRECISION,
        tempo_deslocamento DOUBLE PRECISION,
        possui_apoio_saude_mental TEXT,
        pontuacao_suporte_gestor DOUBLE PRECISION,
        possui_acesso_terapia TEXT,
        dias_afastamento_saude_mental BIGINT,
        faixa_salarial TEXT,
        pontuacao_equilibrio_vida_trabalho DOUBLE PRECISION,
        tamanho_equipe BIGINT,
        pontuacao_crescimento_carreira DOUBLE PRECISION,
        risco_burnout BIGINT
        """,
    )

    # Inserção com upsert, garantindo que id_funcionario não se duplique
    insert_df(conn, df_silver, "silver_employees", conflict_cols=["id_funcionario"])

    cur.close()
    conn.close()
    print(f"[silver] processed {df_silver.height} rows into silver_employees")


if __name__ == "__main__":
    main()
