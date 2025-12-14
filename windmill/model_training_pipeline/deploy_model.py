# service.py
import os
import typing as t
import numpy as np
import pandas as pd
import bentoml
from bentoml.io import JSON

# Nome do modelo no store do BentoML
BENTO_MODEL_NAME = os.getenv("BENTO_MODEL_NAME", "burnout_model")

# Monta tag no formato "<nome>:latest"
model_tag = f"{BENTO_MODEL_NAME}:latest"
print("[service] loading model tag:", model_tag)

# Primeiro tenta obter via API do sklearn (caso modelo tenha sido importado de MLflow como sklearn)
try:
    sklearn_module = bentoml.sklearn
    model_runner = sklearn_module.get(model_tag).to_runner()
except Exception as e:
    # Fallback: acesso direto pelo bentoml.get
    print("[service] sklearn.get failed:", e)
    model_runner = bentoml.get(model_tag).to_runner()

# Define serviço BentoML com o runner do modelo
svc = bentoml.Service(name="burnout_service", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def predict(payload: t.Dict) -> t.Dict:
    """
    Endpoint de previsão. Espera um payload contendo uma lista de instâncias.

    Formato esperado:
    {
      "instances": [
         {"WorkHoursPerWeek": 40, "SleepHours":"6-8", "Gender":"Masculino", "CommuteTime":"Curto", "HasMentalHealthSupport":1},
         ...
      ]
    }

    Retorno:
    {
      "predictions": [0,1,...],
      "probabilities": [0.12, 0.76, ...]
    }
    """
    # Recupera lista de registros enviados para inferência
    instances = payload.get("instances")
    if instances is None:
        return {"error":"payload must contain 'instances' key with a list of input dicts."}

    # Converte para DataFrame, pois o modelo espera formato tabular
    df = pd.DataFrame(instances)

    # Executa predict_proba no runner. Resultado é matriz Nx2 (classe 0, classe 1)
    preds_prob = model_runner.predict_proba.run(df)

    # Extrai probabilidade da classe positiva
    probs = [float(p[1]) if len(p) > 1 else float(p) for p in preds_prob]

    # Classificação usando threshold padrão de 0.5
    preds = [1 if p >= 0.5 else 0 for p in probs]

    return {"predictions": preds, "probabilities": probs}
