# DS-TCC: Contexto e Progresso

## Data de Criação
22 de julho de 2026

## Contexto Inicial

Este projeto DS-TCC é uma evolução do trabalho desenvolvido na disciplina de **Modelos de Machine Learning** do MBA em Ciência de Dados - Unifor. O projeto original está no repositório `ML` (branch `modelagem`) e foi apresentado como trabalho final da disciplina.

### Projeto Original (ML)
- **Localização**: `C:\Unifor\ML\Trabalho_ML\ML`
- **Branch**: `modelagem`
- **Tag de preservação**: `v1.0-apresentado`
- **Objetivo**: Previsão de demanda diária do Top SKU de um supermercado
- **Modelo vencedor**: Extra Trees com MAPE de 12.58%

### Equipe
- **Autores**: Jarlan Lima e Julia Nogueira
- **Orientador**: Prof. Caio Ponte
- **Curso**: MBA em Ciência de Dados - Unifor

---

## Análise Crítica Realizada

### Problemas Identificados no Projeto Original

| # | Problema | Severidade | Status |
|---|----------|------------|--------|
| 1 | **Data leakage no scaler** - StandardScaler era fitado em todos os dados antes do split temporal | Alta | ✓ Corrigido |
| 2 | **Branch main desatualizada** - Todo trabalho estava na branch modelagem | Alta | ✓ Preservado com tag |
| 3 | **requirements.txt com typo e incompleto** - Faltavam scikit-learn e joblib | Média | ✓ Corrigido |
| 4 | **CV do GridSearchCV não era temporal** - cv=3 usava KFold padrão | Média | ✓ Corrigido |
| 5 | **Arquivos de dados no git** - .parquet e .pkl trackeados | Média | ✓ Adicionado ao .gitignore |

### Sugestões do Professor Caio Ponte

O professor forneceu as seguintes anotações durante o desenvolvimento:

1. **"Verificar a possibilidade do modelo com target, qtd e valor"**
   - Decisão: Manter apenas quantidade como target (suficiente para gestão de estoque)
   - Justificativa: Valor agregaria complexidade sem ganho proporcional para o objetivo definido

2. **"Variar também com dias (x1, x2, x3) e valor futuro (x4)"**
   - Implementação: Experimentos com diferentes configurações de lags (3, 7, 15)
   - Status: ✓ Implementado em `src/features.py`

3. **"Verificar a recursiva também, onde a predição fica alimentando para a predição dos outros dias"**
   - Implementação: Função `previsao_recursiva()` em `src/modeling.py`
   - Status: ✓ Implementado no notebook `03_modeling.ipynb`

4. **"Considerar a variação dos Hiperparâmetros W=3, W=7 e W=15"**
   - Implementação: Função `preparar_features_multiplas_janelas()` em `src/features.py`
   - Status: ✓ Implementado no notebook `02_preprocessing.ipynb`

5. **"Verificar multioutputreg e mimo"**
   - Implementação: Função `treinar_multioutput()` em `src/modeling.py`
   - Status: ✓ Implementado no notebook `03_modeling.ipynb`

---

## Decisões de Arquitetura

### 1. Separação de Projetos
- **ML (original)**: Preservado com tag `v1.0-apresentado` para referência
- **DS-TCC (novo)**: Projeto independente com melhorias incrementais

### 2. Estrutura de Diretórios
```
DS-TCC/
├── data/
│   ├── raw/              # Dados brutos (.txt do SysPDV)
│   └── processed/        # Dados processados (.parquet)
├── notebooks/
│   ├── 01_eda.ipynb      # Análise exploratória
│   ├── 02_preprocessing.ipynb  # Engenharia de features (melhorado)
│   └── 03_modeling.ipynb # Treinamento e avaliação (melhorado)
├── models/               # Modelos treinados (.pkl)
├── src/                  # Código modularizado
│   ├── __init__.py
│   ├── data_loader.py    # Carregamento e parsing de dados
│   ├── features.py       # Engenharia de features
│   └── modeling.py       # Modelos e avaliação
├── venv/                 # Ambiente virtual
├── .gitignore
├── requirements.txt
└── README.md
```

### 3. Modularização
O código foi organizado em módulos reutilizáveis:

**data_loader.py:**
- `processar_syspdv_otimizado()`: Parser de arquivos .txt do SysPDV
- `carregar_dados_brutos()`: Carrega todos os arquivos .txt
- `salvar_parquet()`: Salva DataFrame em Parquet
- `carregar_parquet()`: Carrega DataFrame de Parquet

**features.py:**
- `criar_lags()`: Cria features de lag
- `criar_rolling_features()`: Cria média móvel e desvio padrão
- `criar_features_temporais()`: Cria features de data (dia_semana, mes, etc.)
- `preparar_features()`: Pipeline completo de features
- `preparar_features_multiplas_janelas()`: Gera múltiplas configurações (W=3,7,15)
- `criar_features_multi_horizonte()`: Prepara dados para MIMO

**modeling.py:**
- `calcular_metricas()`: MAE, RMSE, MAPE
- `baseline_naive()`: Persistence model
- `treinar_modelo()`: Treina modelos com GridSearchCV
- `previsao_recursiva()`: Multi-step forecasting recursivo
- `treinar_multioutput()`: MultiOutputRegressor (MIMO)
- `despadronizar()`: Reverte StandardScaler
- `salvar_modelo()`, `carregar_modelo()`: Persistência

---

## Melhorias Implementadas

### 1. Correção de Data Leakage (Crítico)
**Problema**: Scaler era fitado em todos os dados antes do split temporal
**Solução**: Scaler é fitado APENAS nos dados de treino

```python
# Antes (incorreto)
scaler.fit_transform(df_todas_as_colunas)
df_train, df_test = split(df)

# Depois (correto)
df_train, df_test = split(df)
scaler.fit(df_train)
df_train_scaled = scaler.transform(df_train)
df_test_scaled = scaler.transform(df_test)
```

**Impacto**: Métricas agora refletem performance real, sem vazamento de informação futura.

### 2. TimeSeriesSplit no GridSearchCV
**Problema**: `cv=3` usava KFold padrão, que pode leakar dados futuros
**Solução**: Usar `TimeSeriesSplit` para validação cruzada temporal

```python
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

tscv = TimeSeriesSplit(n_splits=3)
grid = GridSearchCV(modelo, param_grid, cv=tscv, scoring='neg_mean_absolute_error')
```

### 3. Experimentos com Diferentes Janelas
**Configurações testadas:**
- (3, 3): Lags curtos, janela curta
- (7, 7): Configuração padrão
- (15, 15): Lags longos, janela longa
- (7, 3): Lags padrão, janela curta
- (7, 15): Lags padrão, janela longa
- (3, 7): Lags curtos, janela padrão

**Status**: Dados preparados em `02_preprocessing.ipynb`, avaliação em `03_modeling.ipynb`

### 4. Predição Recursiva Multi-Step
**Abordagem**: Previsão do dia t alimenta previsão do dia t+1

```python
def previsao_recursiva(modelo, X_inicial, steps=7):
    X = X_inicial.copy()
    previsoes = []
    for _ in range(steps):
        pred = modelo.predict(X)[0]
        previsoes.append(pred)
        # Atualiza lags com previsão
        X_novo = X.copy()
        for col in X.columns:
            if col.startswith('lag_'):
                lag_num = int(col.split('_')[1])
                if lag_num == 1:
                    X_novo[col] = pred
                else:
                    X_novo[col] = X[f'lag_{lag_num - 1}']
        X = X_novo
    return np.array(previsoes)
```

**Desafio**: Erro acumula ao longo do horizonte (analisado no notebook)

### 5. MultiOutputRegressor (MIMO)
**Abordagem**: Prever 7 dias simultaneamente em um único modelo

```python
from sklearn.multioutput import MultiOutputRegressor

modelo_base = ExtraTreesRegressor(n_estimators=100, random_state=42)
modelo_multi = MultiOutputRegressor(modelo_base)
modelo_multi.fit(X_train, y_train_multi)  # y_train_multi tem shape (n_samples, 7)
```

**Vantagem**: Captura correlações entre dias consecutivos

### 6. Baseline Naive
**Modelo**: Previsão = valor de ontem (persistence model)
**Propósito**: Referência mínima para avaliar se modelos complexos agregam valor

---

## Ambiente de Execução

### Ambiente Virtual
- **Localização**: `C:\Unifor\ML\DS-TCC\venv`
- **Python**: 3.12.0
- **Kernel Jupyter**: "Python (DS-TCC)"

### Dependências Instaladas
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
joblib>=1.3.0
pyarrow>=12.0.0
fastparquet>=2023.7.0
ipykernel>=6.25.0
```

### Como Executar
```bash
cd C:\Unifor\ML\DS-TCC
.\venv\Scripts\activate
jupyter notebook
```

Executar notebooks na ordem:
1. `01_eda.ipynb` - Análise exploratória
2. `02_preprocessing.ipynb` - Pré-processamento
3. `03_modeling.ipynb` - Modelagem

---

## Status Atual

### Concluído
- [x] Tag criada no projeto original (v1.0-apresentado)
- [x] Estrutura de diretórios criada
- [x] Arquivos copiados e reorganizados
- [x] Módulos de código criados (src/)
- [x] Notebooks reescritos com melhorias
- [x] Ambiente virtual configurado
- [x] Dependências instaladas
- [x] Primeiro commit realizado

### Pendente (Para Continuidade)
- [ ] Executar notebooks e validar resultados
- [ ] Analisar comparação de janelas (W=3, 7, 15)
- [ ] Comparar abordagem recursiva vs MIMO
- [ ] Documentar resultados no README
- [ ] Criar visualizações finais para o TCC
- [ ] Escrever capítulo do TCC com metodologia e resultados

---

## Próximos Passos Recomendados

### 1. Validação dos Resultados
Executar os notebooks e verificar:
- Se as métricas do baseline naive são inferiores aos modelos complexos
- Qual configuração de janela (W) apresenta melhor performance
- Se a abordagem MIMO supera a recursiva
- Como o erro acumula na predição recursiva

### 2. Análise Comparativa
Criar tabelas e gráficos comparando:
- Performance por configuração de janela
- One-step vs recursivo vs MIMO
- Feature importance dos diferentes modelos
- Análise de resíduos (normalidade, heterocedasticidade)

### 3. Documentação para o TCC
Estrutura sugerida:
- **Capítulo 3 - Metodologia**
  - 3.1 Correção de Data Leakage
  - 3.2 Experimentos com Janelas Temporais
  - 3.3 Abordagens Multi-Step (Recursivo e MIMO)
- **Capítulo 4 - Resultados**
  - 4.1 Comparação de Modelos
  - 4.2 Análise de Sensibilidade (Janelas)
  - 4.3 Predição Recursiva vs MIMO
- **Capítulo 5 - Conclusões**
  - 5.1 Lições Aprendidas
  - 5.2 Trabalhos Futuros

### 4. Melhorias Adicionais (Opcional)
- Adicionar features de feriados
- Implementar decomposição STL
- Testar modelos específicos para séries temporais (SARIMA, Prophet)
- Adicionar intervalos de confiança nas previsões

---

## Referências Úteis

### Para Retornar ao Estado Original
```bash
cd C:\Unifor\ML\Trabalho_ML\ML
git checkout v1.0-apresentado
```

### Para Continuar o DS-TCC
```bash
cd C:\Unifor\ML\DS-TCC
.\venv\Scripts\activate
jupyter notebook
```

### Git e Versionamento
- **Repositório original**: `C:\Unifor\ML\Trabalho_ML\ML` (tag: v1.0-apresentado)
- **Repositório novo**: `C:\Unifor\ML\DS-TCC` (branch: master)

---

## Observações Importantes

1. **Dados**: Os arquivos .parquet estão em `data/processed/` e não são versionados (estão no .gitignore)
2. **Modelos**: Os arquivos .pkl estão em `models/` e também não são versionados
3. **Notebooks**: Os notebooks foram reescritos para usar os módulos em `src/`
4. **Caminhos**: Os notebooks usam caminhos relativos (`../data/`, `../models/`)
5. **Kernel**: O kernel "Python (DS-TCC)" já está instalado e configurado

---

## Contato e Suporte

Para dúvidas sobre o projeto original (ML), consultar:
- Tag `v1.0-apresentado` no repositório `ML`
- Notebooks originais: `Eda.ipynb`, `preProcessamento.ipynb`, `modelagem.ipynb`

Para o DS-TCC, este documento serve como ponto de partida.

---

**Última atualização**: 22 de julho de 2026
**Autor**: Jarlan Lima
