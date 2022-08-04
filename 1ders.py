import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys


df=pd.read_csv("C:/Users/ege_s/Documents/Yeni klasör/music_genre.csv")
target="music_genre"

target_values=list(df[target].unique())

del df["instance_id"]

for c in (list(df.columns)):
    if str(df.dtypes[c]) in ["int64","float64"] or c == target:
        pass
    else:
        del df[c]
        
        

clfs={}
for i in target_values:
    clf=RandomForestClassifier(max_depth=5,random_state=0)
    df2=df.copy()
    print(i)
    df2[target]=df2[target].apply(lambda x: 1 if x==i else 0)
    y=df2[target]
    del df2[target] 
    df2=df2.fillna(0)
    clf.fit(df2,y)       
    
    clfs[i]=clf
    
print(clfs)    

y=df[target]
del df[target]
df=df.fillna(0)

preds={}

for key,value in clfs.items():
    preds[key]=clfs[key].predict(df)
    
for key,value in preds.items():
    df[key]=value    
    
df.to_csv("predictions.csv")    


