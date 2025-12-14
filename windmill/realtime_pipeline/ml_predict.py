"""
Lê ./data/pulsar_batch.jsonl, aplica normalização simples e envia para o endpoint REST do BentoML.

Configurações via ENV:
  BENTO_PREDICT_URL (default http://localhost:3000/predict)
  PIPELINE_OUT_DIR  (default ./data)
  PREDICT_BATCH     (tamanho dos batches HTTP, default 100)
"""
import os
import json
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd

OUT_DIR = Path(os.getenv("PIPELINE_OUT_DIR", "./data"))
IN_FILE = OUT_DIR / "pulsar_batch.jsonl"
BENTO_URL = os.getenv("BENTO_PREDICT_URL", "http://localhost:3000/predict")
PREDICT_BATCH = int(os.getenv("PREDICT_BATCH", "100"))

if not IN_FILE.exists():
    raise SystemExit(f"[pulsar_predict] arquivo não encontrado: {IN_FILE} — rode pulsar_get_data.py primeiro")

def load_jsonl(path):
    """
    Carrega JSONL em DataFrame Pandas.
    Cada linha é um JSON independente.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                # Linha inválida é preservada como texto cru
                rows.append({"_raw": line.strip()})
    return pd.DataFrame(rows)

def normalize_row_for_model(df):
    """
    Normalização leve para alinhar as colunas recebidas do Pulsar
    com o formato esperado pelo modelo BentoML.
    Esta etapa pode ser expandida conforme a necessidade do seu modelo.
    """
    col_map = {
        "horas_trabalhadas_semana": "WorkHoursPerWeek",
        "horas_sono": "SleepHours",
        "genero": "Gender",
        "tempo_deslocamento": "CommuteTime",
        "possui_apoio_saude_mental": "HasMentalHealthSupport",
        "id_funcionario": "id_funcionario",

        # Permite também inputs já padronizados
        "WorkHoursPerWeek": "WorkHoursPerWeek",
        "SleepHours": "SleepHours",
        "Gender": "Gender",
        "CommuteTime": "CommuteTime",
        "HasMentalHealthSupport": "HasMentalHealthSupport",
    }

    # Renomeia somente as colunas existentes
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Converte campo de apoio psicológico (Sim/Não, yes/no) para 1/0
    if "HasMentalHealthSupport" in df.columns:
        df["HasMentalHealthSupport"] = (
            df["HasMentalHealthSupport"]
            .replace({"Sim": 1, "Não": 0, "Nao": 0, "yes": 1, "no": 0})
            .fillna(0)
            .astype(int)
        )

    return df

def call_bento(instances):
    """
    Envia um batch para o BentoML.
    Espera resposta contendo:
      { "predictions": [...], "probabilities": [...] }
    """
    payload = {"instances": instances}
    resp = requests.post(BENTO_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

def main():
    # Carrega mensagens consumidas do Pulsar
    df = load_jsonl(IN_FILE)
    if df.empty:
        print("[pulsar_predict] nenhum dado no arquivo, abortando.")
        return

    # Normalização para adequação ao modelo
    df = normalize_row_for_model(df)
    # Guardar features originais dentro do CSV final
    df["_raw_features"] = df.apply(lambda r: r.to_dict(), axis=1)

    # Features esperadas pelo modelo
    model_input_cols = [
        "WorkHoursPerWeek",
        "SleepHours",
        "Gender",
        "CommuteTime",
        "HasMentalHealthSupport"
    ]

    # Seleciona somente as features realmente presentes
    instances = df[[c for c in model_input_cols if c in df.columns]].to_dict(orient="records")

    preds = []
    probs = []
    batch_size = PREDICT_BATCH

    # Envio em batches para o modelo BentoML
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i + batch_size]
        try:
            res = call_bento(batch)
            batch_preds = res.get("predictions", [])
            batch_probs = res.get("probabilities", [])

            # Caso o modelo retorne listas padrão
            if isinstance(batch_probs, list) and batch_preds:
                preds.extend(batch_preds)
                probs.extend(batch_probs)
            else:
                # Fallback para formatos alternativos retornados por alguns modelos
                for item in res.get("items", []):
                    preds.append(item.get("prediction"))
                    probs.append(item.get("probability"))

        except Exception as e:
            print("[pulsar_predict] erro ao chamar BentoML:", e)
            # Em caso de erro no batch, preenche com None
            preds.extend([None] * len(batch))
            probs.extend([None] * len(batch))

    # Garante alinhamento entre previsões e número de registros
    n = len(df)
    preds = (preds + [None] * n)[:n]
    probs = (probs + [None] * n)[:n]

    df["prediction"] = preds
    df["probability"] = probs
    df["predicted_at"] = datetime.utcnow().isoformat()

    # Salva CSV para o step save_ml_predicts
    out_csv = OUT_DIR / f"predictions_{int(datetime.utcnow().timestamp())}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[pulsar_predict] salvo predictions em {out_csv} com {len(df)} linhas")

if __name__ == "__main__":
    main()
