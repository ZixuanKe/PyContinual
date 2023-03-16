import pandas as pd

"""
In this algorithm i´m using python 3.11 cause it´s 80% faster than previous versions 
"""
# Reading dataset
df = pd.read_excel("Rest_Mex_Sentiment_Analysis_2023_Train.xlsx")

# Filter
with open("classified\\hotel.json", "w", encoding='utf-8') as file:
    file.write(df[df["Type"] == "Hotel"].to_json(force_ascii=False, orient='index'))
with open("classified\\restaurant.json", "w", encoding='utf-8') as file:
    file.write(df[df["Type"] == "Restaurant"].to_json(force_ascii=False, orient='index'))
with open("classified\\attractive.json", "w", encoding='utf-8') as file:
    file.write(df[df["Type"] == "Attractive"].to_json(force_ascii=False, orient='index'))
