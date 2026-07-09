import pandas as pd
import numpy as np


def criar_lags(df, coluna='Quantidade', max_lag=7):
    """
    Cria features de lag para a coluna especificada.
    
    Args:
        df: DataFrame com dados diários
        coluna: Nome da coluna para criar lags
        max_lag: Número máximo de lags a criar
        
    Returns:
        DataFrame com colunas de lag adicionadas
    """
    for i in range(1, max_lag + 1):
        df[f'lag_{i}'] = df[coluna].shift(i)
    return df


def criar_rolling_features(df, coluna='Quantidade', janela=7):
    """
    Cria features de rolling (média móvel e desvio padrão).
    
    Args:
        df: DataFrame com dados diários
        coluna: Nome da coluna para criar features
        janela: Tamanho da janela rolling
        
    Returns:
        DataFrame com features rolling adicionadas
    """
    df[f'media_{janela}'] = df[coluna].shift(1).rolling(janela).mean()
    df[f'std_{janela}'] = df[coluna].rolling(janela).std()
    return df


def criar_features_temporais(df, coluna_data='Data'):
    """
    Cria features temporais a partir da coluna de data.
    
    Args:
        df: DataFrame com coluna de data
        coluna_data: Nome da coluna de data
        
    Returns:
        DataFrame com features temporais adicionadas
    """
    df['dia_semana'] = df[coluna_data].dt.dayofweek
    df['mes'] = df[coluna_data].dt.month
    df['dia_mes'] = df[coluna_data].dt.day
    df['fim_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    return df


def preparar_features(df, max_lag=7, janela_rolling=7):
    """
    Pipeline completo de criação de features.
    
    Args:
        df: DataFrame com dados diários (colunas: Data, Quantidade)
        max_lag: Número máximo de lags
        janela_rolling: Tamanho da janela para rolling features
        
    Returns:
        DataFrame com todas as features criadas
    """
    df = df.copy()
    
    df = criar_lags(df, coluna='Quantidade', max_lag=max_lag)
    df = criar_rolling_features(df, coluna='Quantidade', janela=janela_rolling)
    df = criar_features_temporais(df, coluna_data='Data')
    
    df = df.dropna()
    
    return df


def preparar_features_multiplas_janelas(df, configuracoes):
    """
    Cria múltiplas versões de features com diferentes configurações de janelas.
    
    Args:
        df: DataFrame base com dados diários
        configuracoes: Lista de tuplas (max_lag, janela_rolling)
                       Ex: [(3, 3), (7, 7), (15, 15)]
        
    Returns:
        Dicionário com DataFrames para cada configuração
    """
    resultados = {}
    
    for max_lag, janela in configuracoes:
        nome_config = f'lag{max_lag}_w{janela}'
        df_config = preparar_features(df, max_lag=max_lag, janela_rolling=janela)
        resultados[nome_config] = df_config
        print(f"Configuração {nome_config}: {df_config.shape}")
    
    return resultados


def criar_features_multi_horizonte(df, horizonte=7, max_lag=7, janela_rolling=7):
    """
    Prepara dados para abordagem MIMO (Multi-Input Multi-Output).
    Cria targets para múltiplos dias à frente.
    
    Args:
        df: DataFrame com dados diários
        horizonte: Número de dias à frente para prever
        max_lag: Número máximo de lags
        janela_rolling: Tamanho da janela rolling
        
    Returns:
        DataFrame com targets para cada dia do horizonte
    """
    df = df.copy()
    
    df = criar_lags(df, coluna='Quantidade', max_lag=max_lag)
    df = criar_rolling_features(df, coluna='Quantidade', janela=janela_rolling)
    df = criar_features_temporais(df, coluna_data='Data')
    
    for i in range(1, horizonte + 1):
        df[f'target_d{i}'] = df['Quantidade'].shift(-i)
    
    df = df.dropna()
    
    return df
