import pandas as pd

df2022 = pd.read_excel('Track_Train.xlsx')
df2023 =  pd.read_excel('Rest_Mex_Sentiment_Analysis_2023_Train.xlsx')

df = pd.concat([df2023, df2022]).drop_duplicates(keep=False)

df = df[~df.astype(str).apply(lambda x: x.str.contains('Attractive')).any(axis=1)]

df.to_excel('dataset_filtrado.xlsx', index=False)