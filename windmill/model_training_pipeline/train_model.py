# train_model.py
import os
import io
import json
import joblib
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import polars as pl

# Importações essenciais para o pipeline de ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

import mlflow
import mlflow.sklearn
import bentoml

# utils
from u.youruser.utils import get_postgres_conn, read_table_df

# Diretório onde todos os artefatos serão salvos (ROC, CSV de decil, modelo fallback etc.)
OUT_DIR = Path(os.getenv("PIPELINE_OUT_DIR", "./data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nome do modelo no BentoML e parâmetros opcionais da pipeline
BENTO_MODEL_NAME = os.getenv("BENTO_MODEL_NAME", "burnout_model")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "burnout_experiment")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.3"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Mapeamento das colunas da camada silver para nomes usados no notebook/modelo
COLUMN_MAP = {
    "horas_trabalhadas_semana": "WorkHoursPerWeek",
    "horas_sono": "SleepHours",
    "genero": "Gender",
    "tempo_deslocamento": "CommuteTime",
    "possui_apoio_saude_mental": "HasMentalHealthSupport",
    "risco_burnout": "Burnout"
}

def load_data():
    # Prioriza carregar o CSV gerado pela etapa anterior (get_data).
    csv_path = OUT_DIR / "silver_employees.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            print(f"[train] loaded CSV {csv_path} with {len(df)} rows")
            return df
        except Exception as e:
            print("[train] failed to load CSV:", e)

    # Fallback caso o CSV não exista: lê direto do Postgres.
    try:
        conn = get_postgres_conn()
        df_pl = read_table_df(conn, "silver_employees")
        conn.close()

        if df_pl is None or getattr(df_pl, "is_empty", lambda: False)():
            return pd.DataFrame()

        # Converte Polars -> Pandas caso necessário
        df = pl.DataFrame(df_pl).to_pandas() if not isinstance(df_pl, pd.DataFrame) else df_pl
        print(f"[train] loaded table silver_employees with {len(df)} rows")
        return df
    except Exception as e:
        print("[train] failed to read from DB:", e)
        return pd.DataFrame()

def prepare_features(df_raw: pd.DataFrame):
    # Mapeia colunas PT -> EN somente se existirem
    df = df_raw.copy()
    for pt_col, en_col in COLUMN_MAP.items():
        if pt_col in df.columns and en_col not in df.columns:
            df[en_col] = df[pt_col]

    # Sem target Burnout não há supervisão, cai para dataset sintético
    if "Burnout" not in df.columns:
        print("[train] 'Burnout' column not found in silver -> generating fallback synthetic dataset")
        return None

    # Bucketização de horas de sono para categorizar ranges fixos
    if "SleepHours" in df.columns and pd.api.types.is_numeric_dtype(df["SleepHours"]):
        s = df["SleepHours"].fillna(df["SleepHours"].median())
        df["SleepHours"] = pd.cut(
            s,
            bins=[-np.inf, 6, 8, np.inf],
            labels=["<6", "6-8", ">8"]
        )

    # Normalização dos valores de apoio à saúde mental para 0/1
    if "HasMentalHealthSupport" in df.columns:
        df["HasMentalHealthSupport"] = (
            df["HasMentalHealthSupport"]
            .replace({"Sim":1,"Não":0,"Nao":0,"yes":1,"no":0})
            .fillna(0)
            .astype(int)
        )

    # Seleção dinâmica das features presentes
    numerical_features = []
    if "WorkHoursPerWeek" in df.columns:
        numerical_features.append("WorkHoursPerWeek")

    categorical_features = [
        c for c in ["SleepHours","Gender","CommuteTime","HasMentalHealthSupport"]
        if c in df.columns
    ]

    # Filtra somente as colunas úteis para o modelo
    use_cols = numerical_features + categorical_features + ["Burnout"]
    df = df[[c for c in use_cols if c in df.columns]]

    # Remove nulos restantes
    df = df.dropna().reset_index(drop=True)

    # Dataset muito pequeno aciona fallback sintético
    if df.shape[0] < 10:
        print("[train] poucos registros após limpeza -> fallback para dados simulados")
        return None

    return df, numerical_features, categorical_features

def synthetic_dataset(n=1000):
    # Gera dataset aleatório para evitar falhas caso a silver não tenha informações suficientes
    np.random.seed(RANDOM_STATE)
    data = {
        'WorkHoursPerWeek': np.random.randint(30, 70, n),
        'SleepHours': np.random.choice(['<6','6-8','>8'], n, p=[0.3,0.5,0.2]),
        'Gender': np.random.choice(['Masculino','Feminino','Outro'], n, p=[0.45,0.45,0.1]),
        'CommuteTime': np.random.choice(['Curto','Médio','Longo'], n, p=[0.4,0.3,0.3]),
        'HasMentalHealthSupport': np.random.choice([0,1], n, p=[0.6,0.4]),
        'Burnout': np.random.choice([0,1], n, p=[0.75,0.25])
    }
    return pd.DataFrame(data)

def build_pipeline(numerical_features, categorical_features):
    # Pré-processamento: padronização para numéricos e one-hot para categóricos
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    # ColumnTransformer monta um processador automático com base nas colunas recebidas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features) if numerical_features else (),
            ('cat', categorical_transformer, categorical_features) if categorical_features else (),
        ],
        remainder='passthrough'
    )

    # Pipeline unifica pré-processamento e modelo
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=RANDOM_STATE))
    ])
    return model_pipeline

def plot_and_save_roc(y_test, y_pred_proba, out_path: Path):
    # Gera curva ROC e salva imagem em disco
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.4f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return roc_auc

def main():
    # Configuração do MLflow (local ou remoto)
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Carrega dataset
    raw = load_data()
    prepared = prepare_features(raw)

    # Caso o prepare_features retorne None, gera dataset sintético
    if prepared is None:
        df = synthetic_dataset(n=1000)
        numerical_features = ['WorkHoursPerWeek']
        categorical_features = ['SleepHours','Gender','CommuteTime','HasMentalHealthSupport']
    else:
        df, numerical_features, categorical_features = prepared

    # Split em features e target
    X = df.drop(columns=["Burnout"])
    y = df["Burnout"].astype(int)

    # Divide dataset em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Cria pipeline de ML
    model_pipeline = build_pipeline(numerical_features, categorical_features)

    # Grupo MLflow
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print("[mlflow] run id:", run_id)

        # Log de parâmetros
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        # Treina modelo
        model_pipeline.fit(X_train, y_train)
        print("[train] model trained")

        # Predições para avaliação
        y_pred_proba = model_pipeline.predict_proba(X_test)[:,1]
        y_pred_class = (y_pred_proba >= 0.5).astype(int)

        # Métricas
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("roc_auc", float(roc_auc))

        # Classification report
        report_text = classification_report(y_test, y_pred_class, target_names=['NoBurnout','Burnout'])
        mlflow.log_text(report_text, "classification_report.txt")

        # ROC
        roc_path = OUT_DIR / f"roc_{run_id}.png"
        plot_and_save_roc(y_test, y_pred_proba, roc_path)
        mlflow.log_artifact(str(roc_path), artifact_path="plots")

        # Análise de decil (10 grupos pelas probabilidades)
        results_df = pd.DataFrame({"True_Burnout": y_test, "Probability_Score": y_pred_proba})
        results_df['Decil'] = pd.qcut(results_df['Probability_Score'], q=10, labels=False, duplicates='drop')
        results_df['Decil'] = results_df['Decil'].apply(lambda x: 10 - x)
        decil_agg = results_df.groupby('Decil').agg(
            Pessoas=('True_Burnout','count'),
            Casos=('True_Burnout','sum'),
            Min_Score=('Probability_Score','min'),
            Max_Score=('Probability_Score','max')
        ).reset_index()
        decil_path = OUT_DIR / f"decil_{run_id}.csv"
        decil_agg.to_csv(decil_path, index=False)
        mlflow.log_artifact(str(decil_path), artifact_path="analysis")

        # Registra modelo no MLflow
        mlflow.sklearn.log_model(model_pipeline, artifact_path="model")
        print("[mlflow] model logged")

        # Importa modelo para o BentoML store
        model_uri = f"runs:/{run_id}/model"
        bentoml.mlflow.import_model(name=BENTO_MODEL_NAME, model_uri=model_uri)
        print(f"[bentoml] imported mlflow model to bentoml store as '{BENTO_MODEL_NAME}'")

        # Salva modelo local como fallback
        fallback_path = OUT_DIR / f"model_{run_id}.pkl"
        joblib.dump(model_pipeline, fallback_path)
        print("[train] fallback model saved to", fallback_path)

        print("[train] done. run_id:", run_id)
        print("[train] mlflow model uri:", model_uri)

if __name__ == "__main__":
    main()
