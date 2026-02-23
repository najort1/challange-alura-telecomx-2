# Challenge Telecom X - Análise de Evasão de Clientes (Parte 2)

Este repositório contém o fluxo completo de Machine Learning para previsão de churn de clientes da Telecom X usando o dataset tratado da Parte 1.

## Estrutura

- data/ : datasets
- notebooks/ : exploração, modelagem e relatório
- src/ : funções reutilizáveis do pipeline
- reports/ : artefatos de análise
- docs/ : material de apoio do projeto

## Como executar

1. Crie e ative um ambiente virtual
2. Instale dependências
3. Abra os notebooks na pasta notebooks/

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

O dataset principal está em data/processed/churn_final.csv e já contém variáveis tratadas, além de churn_binary como alvo.
