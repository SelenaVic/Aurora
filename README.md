# Aurora — Plataforma de Previsão de Burnout no Trabalho

## Visão Geral do Projeto

A **Aurora** é uma plataforma projetada para identificar, de forma preventiva, colaboradores com maior probabilidade de desenvolver **burnout** em uma janela de tempo definida, permitindo que equipes de RH e saúde ocupacional implementem intervenções personalizadas e oportunas.

O repositório integra:

* **Ingestão de eventos em tempo real** (Apache Pulsar)
* **Data lake** (MinIO) para arquivos brutos e artefatos
* **Camadas de dados** (Bronze → Silver → Gold) organizadas em PostgreSQL
* **Orquestração de pipelines** (Windmill)
* **Treinamento e versionamento de modelos** (MLflow)
* **Serviço de inferência** (BentoML)
* **Scripts e notebook para treino, avaliação e deploy** (Jupyter)
* **Arquivo de Dashboard (Power BI .pbix)** para visualização dos dados

O fluxo contempla tanto processamento em batch (treino, avaliação, relatórios) quanto inferência em produção (consumo de tópico → predição → gravação em gold).

---

## Arquitetura (resumo)

### Batch (treinamento / análise)

* Dados consolidados na camada Gold alimentam notebooks e pipelines de treino.
* Modelos são registrados e versionados no MLflow.
* Artefatos (plots, modelos) armazenados em MinIO.

### Real-time (inferência)

* Eventos chegam via Pulsar.
* Worker consome o tópico, envia lote para o serviço BentoML e grava os resultados em `ml_predicts` (tabela gold).
* Resultados prontas para relatórios e dashboards.

---

## Fundamento e objetivo

**Objetivo:** identificar colaboradores com maior probabilidade de desenvolver burnout em um horizonte futuro definido, suportando ações preventivas personalizadas.

Processo: feature engineering (horas de sono, horas trabalhadas, suporte disponível, indicadores de estresse etc.), treinamento supervisionado e monitoramento de métricas via MLflow.

---

## Estrutura do repositório (corrigida conforme sua imagem)

```
.
├── docker-compose.yml
├── README.md
├── train_model.ipynb
├── aurora_dashboard.pbix      
├── data/                          
├── images/                        
│   ├── powerbi_dashboard.png
│   ├── mlflow_tracking.png
│   ├── windmill_pipeline.png
│   └── email_apoio.png
└── windmill/
    ├── utils.py
    ├── batch_pipeline/
    │   ├── bronze.py
    │   ├── silver.py
    │   └── gold.py
    ├── model_training_pipeline/
    │   ├── get_data.py
    │   ├── train_model.py
    │   └── deploy_model.py
    └── realtime_pipeline/
        ├── get_pulsar_topic.py
        ├── ml_predict.py
        └── save_gold_table.py
```

---

## Arquivos centrais de interesse (caminhos atualizados)

* `windmill/utils.py` — helpers (MinIO, Postgres, Polars).
* `windmill/batch_pipeline/bronze.py` — ingestão e persistência bruta (Bronze).
* `windmill/batch_pipeline/silver.py` — limpeza, renomeação e casts (Bronze → Silver).
* `windmill/batch_pipeline/gold.py` — subset analítico e tabelas finais.
* `windmill/realtime_pipeline/get_pulsar_topic.py` — consumidor Pulsar (obtém mensagens).
* `windmill/realtime_pipeline/ml_predict.py` — chama BentoML para inferência.
* `windmill/realtime_pipeline/save_gold_table.py` — grava predições em `ml_predicts` (Gold).
* `windmill/model_training_pipeline/get_data.py` — exporta silver para CSV (input de treino).
* `windmill/model_training_pipeline/train_model.py` — treino, registro no MLflow, import para BentoML.
* `windmill/model_training_pipeline/deploy_model.py` — passos de deploy (opcional).
* `train_model.ipynb` — notebook com o fluxo de treinamento e análise.
* `aurora_dashboard.pbix` — Power BI Desktop file para visualização.

---

## Dashboards e Interfaces

| Recurso            | Imagem Estática                    |
| ------------------ | ---------------------------------- |
| Power BI Dashboard | ![](/images/powerbi_dashboard.png) |
| MLflow Tracking    | ![](/images/mlflow_tracking.png)   |
| Workflows Windmill | ![](/images/windmill_pipeline.png)     |
| E-mail de Apoio   | ![](/images/email_apoio.png)      |

---

## Tecnologias utilizadas

* **Python** (Polars, Pandas, scikit-learn)
* **BentoML** — serviço de inferência
* **MLflow** — tracking, métricas e artefatos
* **Apache Pulsar** — mensageria em tempo real
* **MinIO** — object storage (S3 compatible)
* **PostgreSQL** — data warehouse das camadas Bronze/Silver/Gold
* **Windmill** — orquestração de steps/pipelines
* **Power BI** — visualização (arquivo `.pbix`)
* **Docker & Docker Compose** — ambiente conteinerizado
