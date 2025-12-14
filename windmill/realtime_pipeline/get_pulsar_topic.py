"""
Consome um lote de mensagens de um tópico Pulsar e salva em ./data/pulsar_batch.jsonl

Configurações via variáveis de ambiente:
  PULSAR_SERVICE_URL (default pulsar://localhost:6650)
  PULSAR_TOPIC       (ex: persistent://public/default/your-topic)
  PULSAR_SUBSCRIPTION(default: pipeline-sub)
  PULSAR_BATCH_SIZE  (default: 200)
  PIPELINE_OUT_DIR   (default ./data)
"""
import os
import json
from pathlib import Path
import time

try:
    import pulsar
except Exception as e:
    raise SystemExit("Instale pulsar-client: pip install pulsar-client") from e

# Diretório onde o batch será salvo
OUT_DIR = Path(os.getenv("PIPELINE_OUT_DIR", "./data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "pulsar_batch.jsonl"

# Configurações do Pulsar
SERVICE_URL = os.getenv("PULSAR_SERVICE_URL", "pulsar://localhost:6650")
TOPIC = os.getenv("PULSAR_TOPIC", "persistent://public/default/your-topic")
SUBSCRIPTION = os.getenv("PULSAR_SUBSCRIPTION", "pipeline-sub")
BATCH_SIZE = int(os.getenv("PULSAR_BATCH_SIZE", "200"))
RECEIVE_TIMEOUT_MS = int(os.getenv("PULSAR_RECEIVE_TIMEOUT_MS", "1500"))

def main():
    # Cria cliente e consumidor Pulsar
    client = pulsar.Client(SERVICE_URL)
    consumer = client.subscribe(TOPIC, SUBSCRIPTION, receiver_queue_size=1000)

    messages = []
    start = time.time()
    print(f"[pulsar_get] connecting to {SERVICE_URL} topic={TOPIC} subscription={SUBSCRIPTION}")
    consumed = 0
    idle_count = 0

    try:
        # Loop até consumir BATCH_SIZE mensagens ou até dar timeout duplo
        while consumed < BATCH_SIZE:
            try:
                # Espera uma mensagem até o timeout configurado
                msg = consumer.receive(timeout_millis="300")
            except pulsar.Timeout:
                idle_count += 1
                # Dois timeouts seguidos indicam que não há mais mensagens recentes
                if idle_count >= 2:
                    print("[pulsar_get] timeout sem mensagens, finalizando batch.")
                    break
                continue

            # Tenta interpretar o payload como JSON
            try:
                data_raw = msg.data().decode("utf-8")
                data = json.loads(data_raw)
            except Exception:
                # Se não for JSON válido, salva como texto cru
                data = {"_raw_message": msg.data().decode("utf-8")}

            messages.append(data)
            consumer.acknowledge(msg)
            consumed += 1

    finally:
        # Fecha recursos
        consumer.close()
        client.close()

    # Salva o batch em formato JSONL para ser processado pelo próximo step
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        for item in messages:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[pulsar_get] gravado {len(messages)} mensagens em {OUT_FILE} (tempo {time.time()-start:.2f}s)")

if __name__ == "__main__":
    main()
