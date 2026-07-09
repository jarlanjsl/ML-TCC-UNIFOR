# DS-TCC: Previsão de Demanda com Séries Temporais

## Descrição

Trabalho de Conclusão de Curso do MBA em Ciência de Dados - Unifor.

Este projeto implementa um sistema preditivo para demanda diária de produtos de supermercado, utilizando técnicas avançadas de séries temporais e machine learning.

## Contexto

O projeto é uma evolução do trabalho desenvolvido na disciplina de **Modelos de Machine Learning**, que previa a demanda do Top SKU usando abordagem básica. Esta versão implementa melhorias significativas solicitadas pelo professor Caio Ponte.

## Melhorias em Relação ao Projeto Anterior (ML)

### 1. Correção de Data Leakage
- **Problema anterior:** `StandardScaler` era fitado em todos os dados antes do split temporal
- **Solução:** Scaler é fitado apenas nos dados de treino, transformando teste separadamente
- **Impacto:** Métricas agora refletem performance real, sem vazamento de informação futura

### 2. Experimentos com Diferentes Janelas (W)
- **W=3:** Captura variações recentes (curto prazo)
- **W=7:** Captura padrão semanal (atual)
- **W=15:** Suaviza ruído (longo prazo)
- **Análise:** Matriz de experimentos comparando combinações de lags e janelas

### 3. Predição Recursiva Multi-Step
- **Abordagem:** Previsão do dia t alimenta previsão do dia t+1
- **Benefício:** Captura dependência temporal entre dias consecutivos
- **Análise:** Visualização do erro acumulado ao longo do horizonte

### 4. MultiOutputRegressor (MIMO)
- **Abordagem:** Prever 7 dias simultaneamente em um único modelo
- **Comparação:** vs. abordagem recursiva
- **Métricas:** Performance por horizonte (dia+1, dia+2, ... dia+7)

### 5. Baseline Naive
- **Modelo:** Previsão = valor de ontem (persistence model)
- **Propósito:** Referência mínima para avaliar se os modelos complexos agregam valor

### 6. Modularização do Código
- **Estrutura:** Código organizado em módulos reutilizáveis (`src/`)
- **Benefícios:** Manutenibilidade, testabilidade e reuso

## Estrutura do Projeto

```
DS-TCC/
├── data/
│   ├── raw/              # Dados brutos (.txt do SysPDV)
│   └── processed/        # Dados processados (.parquet)
├── notebooks/
│   ├── 01_eda.ipynb      # Análise exploratória
│   ├── 02_preprocessing.ipynb  # Engenharia de features
│   └── 03_modeling.ipynb # Treinamento e avaliação
├── models/               # Modelos treinados (.pkl)
├── src/                  # Código modularizado
│   ├── __init__.py
│   ├── data_loader.py    # Carregamento e parsing de dados
│   ├── features.py       # Engenharia de features
│   └── modeling.py       # Modelos e avaliação
├── .gitignore
├── requirements.txt
└── README.md
```

## Instalação

### Pré-requisitos
- Python 3.8+
- pip

### Passos

1. **Clonar o repositório**
```bash
git clone https://github.com/seu-usuario/DS-TCC.git
cd DS-TCC
```

2. **Criar ambiente virtual**
```bash
python -m venv venv
```

3. **Ativar ambiente virtual**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. **Instalar dependências**
```bash
pip install -r requirements.txt
```

5. **Instalar kernel Jupyter**
```bash
python -m ipykernel install --user --name=ds-tcc --display-name="Python (DS-TCC)"
```

## Uso

### Executar Notebooks

1. Ative o ambiente virtual
2. Inicie o Jupyter:
```bash
jupyter notebook
```
3. Navegue até `notebooks/` e execute na ordem:
   - `01_eda.ipynb` - Análise exploratória
   - `02_preprocessing.ipynb` - Pré-processamento
   - `03_modeling.ipynb` - Modelagem

### Estrutura dos Notebooks

**01_eda.ipynb:**
- Extração de dados brutos (.txt)
- Limpeza e filtragem
- Análise de Pareto (volume vs faturamento)
- Estatísticas descritivas
- Visualizações de sazonalidade

**02_preprocessing.ipynb:**
- Seleção do Top SKU
- Criação de lags (1-15 dias)
- Rolling features (média, std) com janelas configuráveis
- Features temporais (dia_semana, mes, fim_de_semana)
- **Correção:** Split temporal ANTES do scaling
- Salvamento do scaler

**03_modeling.ipynb:**
- Baseline naive
- Modelos: Linear Regression, Random Forest, Gradient Boosting, Extra Trees
- GridSearchCV com TimeSeriesSplit
- **Novo:** Predição recursiva multi-step
- **Novo:** MultiOutputRegressor (MIMO)
- Comparação de métricas (MAE, RMSE, MAPE)
- Feature importance
- Análise de resíduos

## Experimentos Planejados

### Matriz de Configurações

| Config | Lags | Rolling W | Modelo | Métrica |
|--------|------|-----------|--------|---------|
| 1 | 1-3 | 3 | Extra Trees | MAPE |
| 2 | 1-7 | 7 | Extra Trees | MAPE |
| 3 | 1-15 | 15 | Extra Trees | MAPE |
| 4 | 1-7 | 3 | Extra Trees | MAPE |
| 5 | 1-7 | 15 | Extra Trees | MAPE |
| 6 | 1-3 | 7 | Extra Trees | MAPE |

### Abordagens de Previsão Multi-Step

1. **One-step ahead:** Prever apenas 1 dia (baseline atual)
2. **Recursivo:** Prever dia t, usar como input para dia t+1
3. **MIMO:** Prever 7 dias simultaneamente

## Resultados Esperados

- Identificar configuração ótima de janelas (W)
- Comparar performance de abordagens recursiva vs MIMO
- Quantificar erro acumulado na predição recursiva
- Estabelecer baseline para comparação
- Documentar lições aprendidas

## Equipe

- **Autor:** Jarlan Lima e Julia Nogueira
- **Orientador:** Prof. Caio Ponte
- **Curso:** MBA em Ciência de Dados - Unifor

## Licença

Este projeto foi desenvolvido para fins acadêmicos.

## Referências

- Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: principles and practice
- Scikit-learn documentation: Time series forecasting
- Pandas documentation: Time series functionality
