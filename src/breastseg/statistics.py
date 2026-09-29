"""Exploratory paired inference; seeds are averaged, never treated as new patients."""
import numpy as np
from scipy.stats import wilcoxon

def bootstrap_ci(values,n=10000,seed=20260929):
    a=np.asarray(values,float);rng=np.random.default_rng(seed)
    if a.ndim!=1 or len(a)<2 or not np.isfinite(a).all(): raise ValueError('Need >=2 finite units')
    samples=a[rng.integers(0,len(a),(n,len(a)))].mean(1)
    return np.percentile(samples,[2.5,97.5]).tolist()

def holm(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p);out=np.empty(len(p));running=0.
    for i,index in enumerate(order):
        running=max(running,(len(p)-i)*p[index]);out[index]=min(1.,running)
    return out.tolist()

def paired(a,b):
    d=np.asarray(a,float)-np.asarray(b,float)
    return dict(mean_difference=float(d.mean()),ci95=bootstrap_ci(d),
        p=float(wilcoxon(d,alternative='two-sided',zero_method='wilcox',method='auto').pvalue) if np.any(d) else 1.)
