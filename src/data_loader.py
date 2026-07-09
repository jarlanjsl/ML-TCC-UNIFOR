import pandas as pd
import glob
import os


def processar_syspdv_otimizado(caminho_arquivo):
    """
    Processa um arquivo .txt do SysPDV e extrai dados estruturados.
    
    Args:
        caminho_arquivo: Caminho para o arquivo .txt
        
    Returns:
        DataFrame com dados extraídos
    """
    layout_essencial = [
        ('Tipo_Registro', 0, 2),
        ('Data', 35, 43),
        ('Hora', 151, 155),
        ('Cod_Produto', 43, 57),
        ('Cod_Secao', 19, 21),
        ('Cod_Loja', 2, 6),
        ('Quantidade', 57, 66),
        ('Valor_Unitario', 66, 78),
        ('Desconto', 78, 90),
        ('Preco_Custo', 194, 206),
        ('Acréscimos', 90, 102),
        ('Preço praticado', 129, 130),
        ('Total', 130, 142),
        ('Preço de Venda', 155, 167),
        ('Tipo de bonificação', 167, 168),
        ('Fator', 168, 177),
        ('Identificação Consumidor', 274, 288),
        ('Operador', 288, 294),
        ('Vendedor', 294, 300),
        ('Cartao_Fidelidade', 226, 245)
    ]

    registros = []
    try:
        with open(caminho_arquivo, 'r', encoding='latin-1') as f:
            for linha in f:
                if linha.startswith('01'):
                    dados_linha = {campo: linha[inicio:fim].strip() for campo, inicio, fim in layout_essencial}
                    registros.append(dados_linha)
        
        df = pd.DataFrame(registros)
        df = df[df['Tipo_Registro'] == '01'].copy()
        
        df['Data'] = pd.to_datetime(df['Data'], format='%d%m%Y', errors='coerce')
        df['Quantidade'] = pd.to_numeric(df['Quantidade'], errors='coerce')
        
        cols_financeiras = ['Valor_Unitario', 'Desconto', 'Preco_Custo', 'Acréscimos', 'Preço praticado', 'Total', 'Preço de Venda']
        for col in cols_financeiras:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        cols_cat = ['Cod_Produto', 'Cod_Secao', 'Cod_Loja']
        for col in cols_cat:
            df[col] = df[col].astype('category')

        return df
    except Exception as e:
        print(f"Erro ao processar {caminho_arquivo}: {e}")
        return pd.DataFrame()


def carregar_dados_brutos(pasta_arquivos='data/raw'):
    """
    Carrega todos os arquivos .txt da pasta especificada.
    
    Args:
        pasta_arquivos: Caminho para a pasta com arquivos .txt
        
    Returns:
        DataFrame consolidado
    """
    lista_arquivos = glob.glob(os.path.join(pasta_arquivos, '*.txt'))
    todos_dfs = [processar_syspdv_otimizado(arq) for arq in lista_arquivos]
    
    if todos_dfs:
        return pd.concat(todos_dfs, ignore_index=True)
    else:
        print("Nenhum dado processado.")
        return pd.DataFrame()


def salvar_parquet(df, caminho='data/processed/vendas_supermercado.parquet'):
    """
    Salva DataFrame em formato Parquet.
    
    Args:
        df: DataFrame a ser salvo
        caminho: Caminho de destino
    """
    df.to_parquet(caminho, index=False)
    print(f"Arquivo Parquet salvo em: {caminho}")


def carregar_parquet(caminho='data/processed/vendas_supermercado.parquet', engine='pyarrow'):
    """
    Carrega DataFrame de arquivo Parquet.
    
    Args:
        caminho: Caminho do arquivo
        engine: Engine para leitura (pyarrow ou fastparquet)
        
    Returns:
        DataFrame carregado
    """
    return pd.read_parquet(caminho, engine=engine)
