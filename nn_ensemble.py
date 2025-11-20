import os
import pandas as pd

cnt = 0.0
score = 0.0

for item in os.listdir('./temp_nn/'):
    score += float(item.split('_')[1])
    tmp = pd.read_csv('./temp_nn/'+item)
    if cnt == 0:
        preds = tmp.copy()
    else:
        preds['prediction'] += tmp['prediction']
    cnt += 1.0

score /= cnt
preds['prediction'] /= cnt
preds.to_csv('./submission/ensemble.csv', index=False)

