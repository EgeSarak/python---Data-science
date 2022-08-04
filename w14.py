
from inspect import CO_COROUTINE
from utility import *
import pandas as pd
import sys
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
df['product_category'] = df['product_category'].astype(str)
df = df.groupby(['customer_id'])['product_category'].apply(lambda x: ';'.join(x)).reset_index()


df = df[ df['product_category'].str.contains(';') ]


import networkx as nx
G = nx.Graph()

co_occourance = {}

for i in range(len(df)):
    row = df.iloc[i].to_dict()
    parts = row['product_category'].split(';')
    for p1 in parts:
        for p2 in parts:
            if p1 > p2:
                key = p1 + "_" + p2
                if key not in co_occourance:
                    co_occourance[key] = 1
                else:
                    co_occourance[key] += 1
                
for k in co_occourance:
    p1, p2 = k.split("_")
    if co_occourance[k] > 60:
        n = (co_occourance[k] - 49) / 5
        G.add_edge(p1, p2, weight=n)


pos = nx.spring_layout(G, seed=63)

nx.draw(G, pos, with_labels = True)


for edge in G.edges(data='weight'):
    nx.draw_networkx_edges(G, pos, edgelist=[edge], width=edge[2])


plt.show()




print(df)
sys.exit(1)

freq = {}
co_occourance = {}

for line in readTextFile2("data.txt"):
    parts = line.split(" ")
    for p in parts:
        if p not in freq:
            freq[p] = 1
        else:
            freq[p] += 1

    for p1 in parts:
        for p2 in parts:
            if p1 > p2:
                key = p1 + "_" + p2
                
                if key not in co_occourance:
                    co_occourance[key] = 1
                else:
                    co_occourance[key] += 1


