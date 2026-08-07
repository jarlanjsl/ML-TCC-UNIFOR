import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib


def calcular_metricas(y_true, y_pred):
    """
    Calcula métricas de regressão.
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        
    Returns:
        Dicionário com MAE, RMSE e MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def baseline_naive(y_train, y_test):
    """
    Modelo baseline naive (persistence model).
    Previsão = valor do dia anterior.
    
    Args:
        y_train: Série temporal de treino
        y_test: Série temporal de teste
        
    Returns:
        Previsões do baseline
    """
    y_pred = y_test.shift(1)
    y_pred = y_pred.dropna()
    y_test_aligned = y_test.loc[y_pred.index]
    
    return y_test_aligned, y_pred


def treinar_modelo(nome_modelo, X_train, y_train, X_test, y_test, usar_gridsearch=False):
    """
    Treina um modelo e calcula métricas.
    
    Args:
        nome_modelo: Nome do modelo ('lr', 'rf', 'gb', 'et')
        X_train: Features de treino
        y_train: Target de treino
        X_test: Features de teste
        y_test: Target de teste
        usar_gridsearch: Se True, usa GridSearchCV para otimização
        
    Returns:
        Tupla (modelo, metricas, y_pred)
    """
    modelos = {
        'lr': LinearRegression(),
        'rf': RandomForestRegressor(random_state=42),
        'gb': GradientBoostingRegressor(random_state=42),
        'et': ExtraTreesRegressor(random_state=42, n_jobs=-1)
    }
    
    if nome_modelo not in modelos:
        raise ValueError(f"Modelo '{nome_modelo}' não suportado. Use: {list(modelos.keys())}")
    
    modelo = modelos[nome_modelo]
    
    if usar_gridsearch and nome_modelo != 'lr':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(
            modelo, 
            param_grid, 
            cv=tscv, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        modelo = grid.best_estimator_
        print(f"Melhores parâmetros: {grid.best_params_}")
    
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    metricas = calcular_metricas(y_test, y_pred)
    
    return modelo, metricas, y_pred


def previsao_recursiva(modelo, X_inicial, steps=7):
    """
    Realiza previsão recursiva multi-step.
    A previsão do dia t é usada como feature para o dia t+1.
    
    Args:
        modelo: Modelo treinado
        X_inicial: Features iniciais (último dia conhecido)
        steps: Número de dias à frente para prever
        
    Returns:
        Array com previsões para cada dia
    """
    X = X_inicial.copy()
    previsoes = []
    
    for _ in range(steps):
        pred = modelo.predict(X)[0]
        previsoes.append(pred)
        
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


def treinar_multioutput(X_train, y_train_multi, X_test, y_test_multi, horizonte=7):
    """
    Treina modelo MultiOutputRegressor para previsão MIMO.
    
    Args:
        X_train: Features de treino
        y_train_multi: Targets para múltiplos dias (shape: n_samples, horizonte)
        X_test: Features de teste
        y_test_multi: Targets de teste para múltiplos dias
        horizonte: Número de dias à frente
        
    Returns:
        Tupla (modelo, metricas_por_horizonte, previsoes)
    """
    modelo_base = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo_multi = MultiOutputRegressor(modelo_base)
    
    modelo_multi.fit(X_train, y_train_multi)
    y_pred_multi = modelo_multi.predict(X_test)
    
    metricas_por_horizonte = []
    for i in range(horizonte):
        metricas = calcular_metricas(
            y_test_multi.iloc[:, i] if hasattr(y_test_multi, 'iloc') else y_test_multi[:, i],
            y_pred_multi[:, i]
        )
        metricas['horizonte'] = f'dia+{i+1}'
        metricas_por_horizonte.append(metricas)
    
    return modelo_multi, metricas_por_horizonte, y_pred_multi


def despadronizar(valores_z, media, escala):
    """
    Reverte a padronização (StandardScaler).
    
    Args:
        valores_z: Valores padronizados
        media: Média usada na padronização
        escala: Escala (std) usada na padronização
        
    Returns:
        Valores na escala original
    """
    return (valores_z * escala) + media


def salvar_modelo(modelo, caminho):
    """
    Salva modelo treinado em arquivo .pkl.
    
    Args:
        modelo: Modelo a ser salvo
        caminho: Caminho de destino
    """
    joblib.dump(modelo, caminho)
    print(f"Modelo salvo em: {caminho}")


def carregar_modelo(caminho):
    """
    Carrega modelo treinado de arquivo .pkl.
    
    Args:
        caminho: Caminho do arquivo
        
    Returns:
        Modelo carregado
    """
    return joblib.load(caminho)


def comparar_modelos(resultados):
    """
    Cria tabela comparativa de métricas entre modelos.
    
    Args:
        resultados: Lista de dicionários com 'nome' e 'metricas'
        
    Returns:
        DataFrame com comparação
    """
    df_comparacao = pd.DataFrame(resultados)
    df_comparacao = df_comparacao.sort_values('MAPE')
    return df_comparacao
