import pandas as pd
import json

df2022 = pd.read_excel('Track_Train.xlsx')
df2023 =  pd.read_excel('Rest_Mex_Sentiment_Analysis_2023_Train.xlsx')

# filas2022, columnas2022 = df2022.shape
# filas2023, columnas2023 = df2023.shape

# print('Dataset de 2022', filas2022, 'filas.')
# print('Dataset de 2023', filas2023, 'filas.')


# Encontrar la intersección
interseccion1 = pd.merge( df2023,df2022, on='Review')
interseccion2 = pd.merge( df2022,df2023, on='Review')

# Saving...
interseccion1.to_excel('interseccion2023-2022.xlsx', index=False)
interseccion2.to_excel('interseccion2022-2023.xlsx', index=False)
    




