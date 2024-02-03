import pandas as pd

# Cargando los datasets
df2022 = pd.read_excel('Track_Train.xlsx')
df2023 =  pd.read_excel('Rest_Mex_Sentiment_Analysis_2023_Train.xlsx')

# Filtrando hasta quedarse con las filas unicas
reviews_ds22 = df2022['Review'].unique().tolist()
reviews_ds23 = df2023['Review'].unique().tolist()

# Esta query permite saber si el contenido de la columna review esta en el dataset 
# con el q se le esta comparando
rows_only_in_ds22_df = df2023[~df2023['Review'].isin(reviews_ds22)]
rows_only_in_ds23_df = df2022[~df2022['Review'].isin(reviews_ds23)]

# Exportando los datasets resultantes
rows_only_in_ds22_df.to_excel('only_2022.xlsx', index=False)
rows_only_in_ds23_df.to_excel('only_2023.xlsx', index=False)