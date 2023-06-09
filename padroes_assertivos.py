import streamlit as st
import pandas as pd
import itertools
from tabulate import tabulate

# Configurações iniciais do Streamlit
st.set_page_config(layout="wide")

# Função para extrair os resultados do primeiro tempo, tempo final e partidas
def extrair_resultados(resultado):
    if resultado != '?\n\n?':
        resultado_split = resultado.split('\n\n')
        primeiro_tempo = resultado_split[1]
        tempo_final = resultado_split[0]
        return primeiro_tempo, tempo_final
    else:
        return None, None

# Função para encontrar as melhores combinações
def find_top_combinations(df):
    num_conjuntos_range = range(1, len(df) + 1)
    combinations = []
    
    for num_conjuntos in num_conjuntos_range:
        possible_combinations = list(itertools.combinations(df.iterrows(), num_conjuntos))
        combinations.extend(possible_combinations)
    
    results = []
    
    for combination in combinations:
        data = []
        num_conjuntos = len(combination)
        
        for _, row in combination:
            primeiro_tempo = row['Primeiro tempo']
            tempo_final = row['Tempo final']
            num_total_partidas = row['Num total partidas']
            
            resultado_analise = analisar_partidas(df, primeiro_tempo, tempo_final, num_total_partidas, num_conjuntos)
            dicionario = criar_novo_dicionario(resultado_analise, num_total_partidas)
            
            num_conjuntos = len(dicionario[1][0])  # Número de valores em cada lista
            num_total = len(resultado_analise)
            
            data = []  # Lista para armazenar os dados das linhas do dataframe
            
            for key, lista_chave in dicionario.items():
                row = [format(key)]
                AM_counts = [0] * num_conjuntos
                AN_counts = [0] * num_conjuntos
                Over_15_counts = [0] * num_conjuntos
                Over_25_counts = [0] * num_conjuntos
                Over_35_counts = [0] * num_conjuntos
                total_AM = 0
                total_AN = 0
                total_over_15 = 0
                total_over_25 = 0
                total_over_35 = 0
                
                for lista in lista_chave:
                    AM_found = False
                    AN_found = False
                    over_15_found = False
                    over_25_found = False
                    over_35_found = False
                    
                    for i, val in enumerate(lista):
                        score1, score2 = val.split('x')
                        score1 = int(score1)
                        score2 = int(score2)
                        
                        if not AM_found and score1 >= 1 and score2 >= 1:
                            AM_counts[i] += 1
                            AM_found = True
                            
                            if score1 + score2 > 1.5 and score1 + score2 < 2.5:
                                Over_15_counts[i] += 1
                                over_15_found = True
                                
                            if score1 + score2 > 2.5 and not over_15_found and score1 + score2 < 3.5:  # Verificar se não foi contado como over 1.5
                                Over_25_counts[i] += 1
                                over_25_found = True
                                
                            if score1 + score2 > 3.5 and not over_15_found and not over_25_found:  # Verificar se não foi contado como over 1.5 e over 2.5
                                Over_35_counts[i] += 1
                                over_35_found = True
                        
                        if not AN_found and (score1 < 1 or score2 < 1):
                            AN_counts[i] += 1
                            AN_found = True
                                                
                    total_AM += int(AM_found)
                    total_AN += int(AN_found)
                    total_over_15 += int(over_15_found)
                    total_over_25 += int(over_25_found)
                    total_over_35 += int(over_35_found)
                    
                row.extend(Over_15_counts)
                row.extend(Over_25_counts)
                row.extend(Over_35_counts)
                row.extend(AM_counts)
                row.extend(AN_counts)
                row.append(sum(Over_15_counts))
                row.append(sum(Over_25_counts))
                row.append(sum(Over_35_counts))
                row.append(sum(AM_counts))
                row.append(sum(AN_counts))
                data.append(row)
            
            columns = ['Partidas após'] + [f'{i} (Over 1.5)' for i in range(1, num_conjuntos+1)] + [f'{i} (Over 2.5)' for i in range(1, num_conjuntos+1)] + [f'{i} (Over 3.5)' for i in range(1, num_conjuntos+1)] + [f'{i} (AM)' for i in range(1, num_conjuntos+1)] + [f'{i} (AN)' for i in range(1, num_conjuntos+1)] + ['Total Over 1.5', 'Total Over 2.5', 'Total Over 3.5', 'Total AM', 'Total AN']
            df = pd.DataFrame(data, columns=columns)
            df.iloc[:, 1:1+num_conjuntos*3] = df.iloc[:, 1:1+num_conjuntos*3].apply(pd.to_numeric)
            df['Total Over 1.5'] = df.iloc[:, 1:1+num_conjuntos].sum(axis=1)
            df['Total Over 2.5'] = df.iloc[:, 1+num_conjuntos:1+2*num_conjuntos].sum(axis=1)
            df['Total Over 3.5'] = df.iloc[:, 1+2*num_conjuntos:1+3*num_conjuntos].sum(axis=1)
            df['Total AM'] = df.iloc[:, 1+3*num_conjuntos:1+4*num_conjuntos].sum(axis=1)
            df['Total AN'] = df.iloc[:, 1+4*num_conjuntos:1+5*num_conjuntos].sum(axis=1)
            
            total_percent = "{:.2%}".format(1 / num_total)
            
            df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: str(x) + f'/{num_total} ({float(x)/num_total:.2%})' if isinstance(x, int) else x)
            
            df = df.sort_values(by=['Total AM', 'Total AN', 'Total Over 1.5', 'Total Over 2.5', 'Total Over 3.5'], ascending=False)
            df = df.reset_index(drop=True)
            
            accuracy = df.loc[0, 'Total AM'] + df.loc[0, 'Total AN'] + df.loc[0, 'Total Over 1.5'] + df.loc[0, 'Total Over 2.5'] + df.loc[0, 'Total Over 3.5']
            
            results.append((accuracy, df))
    
    results.sort(reverse=True)
    
    return results[:10]

# Interface do Streamlit
st.title("Análise de Combinações em Jogos de Futebol")

# Upload do arquivo Excel
file = st.file_uploader("Selecione o arquivo Excel", type=["xlsx"])

if file is not None:
    # Leitura do arquivo Excel
    df = pd.read_excel(file, sheet_name=None)
    
    # Seleção da página
    sheet_names = list(df.keys())
    selected_sheet = st.selectbox("Selecione a página", sheet_names)

    # Exibição do conjunto de dados
    st.subheader(f"Conjunto de Dados - {selected_sheet}")
    df_selected = df[selected_sheet]
    st.write(df_selected)

    # Botão para gerar combinações
    if st.button("Gerar combinações mais assertivas"):
        # Processamento dos dados
        df_processed = process_data(df_selected)

        # Encontrar as melhores combinações
        melhores_combinacoes = find_top_combinations(df_processed)

        # Exibição das melhores combinações
        st.subheader("Melhores Combinações")
        for i, (accuracy, combination_df) in enumerate(melhores_combinacoes):
            st.subheader(f"Combinação {i+1}")
            st.write(combination_df)
            st.write("--------------------------------------------------")
