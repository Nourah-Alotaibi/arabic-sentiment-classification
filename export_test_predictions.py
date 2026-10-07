"""Export predictions in the shared prepared.csv test-row order for paired comparison."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/prepared.csv')
    p.add_argument('--model',default='models/model.joblib')
    p.add_argument('--output',default='data/classical-test-predictions.npy')
    args=p.parse_args()
    data=pd.read_csv(args.data)
    if data.normalized.duplicated().any():raise ValueError('Expected the deduplicated shared dataset.')
    subset=data.loc[data.split.eq('test'),'normalized']
    if subset.empty:raise ValueError('No test rows were found.')
    prediction=joblib.load(args.model).predict(subset).astype(str)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.save(out,prediction,allow_pickle=False)
    print(f'Exported {len(prediction)} predictions in shared test-row order.')
