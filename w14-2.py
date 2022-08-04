

import numpy as np
import pandas as pd
df = pd.read_csv("data.csv")

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df)
tr = pca.transform( df )
re = pca.inverse_transform( tr )



def safe(x, divider):
    if divider == 0: return 0
    return x / divider

re = pd.DataFrame(re)
re.columns = df.columns

for i in range(len(df)):
    orig = df.iloc[i]
    recn = re.iloc[i]

    orig = list(orig)
    recn = list(recn)

    ape = [abs(safe(orig[o] - recn[o],orig[o])) for o in range(len(orig))]
    mape = np.mean(ape)
    if mape > 0.50:
        print(mape, i, orig, recn)






# pca anomaly
# cluster


