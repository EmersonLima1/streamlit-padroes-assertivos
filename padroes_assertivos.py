import streamlit as st
import pandas as pd
from openpyxl import load_workbook

# Carregando o arquivo Excel
uploaded_file = st.file_uploader("Faça upload do arquivo Excel", type="xlsx")

if uploaded_file:
    # Carregar o arquivo Excel usando o openpyxl
    excel_data = pd.ExcelFile(uploaded_file)
    sheet_names = excel_data.sheet_names

    # Definir todas as opções de Primeiro Tempo, Tempo Final, quantidade de entrada e Tip
    primeiro_tempo_options = ['0x0', '0x1', '0x2', '0x3', '0x4', '0x5', '1x0', '1x1', '1x2', '1x3', '1x4', '1x5', '2x1', '2x2', '2x3', '2x4', '2x5', '3x1', '3x2', '3x3', '3x4', '3x5', '4x1', '4x2', '4x3', '4x4', '4x5', '5x1', '5x2', '5x3', '5x4', '5x5']
    tempo_final_options = ['0x0', '0x1', '0x2', '0x3', '0x4', '0x5', '1x0', '1x1', '1x2', '1x3', '1x4', '1x5', '2x1', '2x2', '2x3', '2x4', '2x5', '3x1', '3x2', '3x3', '3x4', '3x5', '4x1', '4x2', '4x3', '4x4', '4x5', '5x1', '5x2', '5x3', '5x4', '5x5']
    num_total_partidas_options = list(range(1, 51))
    num_conjuntos_options = [1, 2, 3, 4, 5]

    # Dicionário para armazenar os resultados das combinações
    resultados = {}
    
    for sheet_name in sheet_names:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        for primeiro_tempo in primeiro_tempo_options:
            for tempo_final in tempo_final_options:
                if primeiro_tempo is None and tempo_final is None:
                    continue
                for num_total_partidas in num_total_partidas_options:
                    for num_conjuntos in num_conjuntos_options:
                        if primeiro_tempo is None and tempo_final is None:
                            continue
                            
                        st.write(primeiro_tempo)
                        st.write(tempo_final)
                        st.write(num_total_partidas)
                        st.write(num_conjuntos)
