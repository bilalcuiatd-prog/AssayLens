from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

BG='#101e29';FG='#e2edf2';ACC='#3dd1b7';SECONDARY='#80aaff'


def make_figure(out, x,y,xlabel,ylabel,title,kind='line',reference=False):
    fig,ax=plt.subplots(figsize=(6.8,4.5))
    fig.patch.set_facecolor(BG);ax.set_facecolor(BG)
    if kind=='scatter':ax.scatter(x,y,c=ACC,s=15,alpha=.5)
    elif kind=='bar':ax.bar(x,y,color=ACC)
    else:ax.plot(x,y,color=ACC,lw=2.2)
    if reference:
        vals=np.asarray(list(x)+list(y),dtype=float)
        lo,hi=np.nanmin(vals),np.nanmax(vals)
        ax.plot([lo,hi],[lo,hi],c=SECONDARY,ls='--',alpha=.7)
    ax.set_title(title,color=FG,pad=13)
    ax.set_xlabel(xlabel,color=FG);ax.set_ylabel(ylabel,color=FG)
    ax.tick_params(colors=FG)
    for side in ax.spines.values():side.set_color('#446071')
    fig.tight_layout();fig.savefig(out,dpi=160,facecolor=BG);plt.close(fig)


def create_diagnostics(folder,task):
    folder=Path(folder)
    preds=pd.read_csv(folder/'out_of_fold_predictions.csv')
    if preds.empty:return []
    first=preds[(preds.seed==preds.seed.min()) & (preds.model==preds.model.iloc[0]) & (preds.ablation==preds.ablation.iloc[0])].copy()
    if first.empty:return []
    files=[]
    if task=='regression':
        make_figure(folder/'observed_vs_predicted.png',first.observed,first.predicted,
                    'Observed pActivity','Predicted pActivity','Held-out predictions · first configured model',kind='scatter',reference=True)
        files.append('observed_vs_predicted.png')
        first['absolute_error']=(first.observed-first.predicted).abs()
        bins=pd.qcut(first.observed,q=min(5,len(first.observed.unique())),duplicates='drop')
        errors=first.groupby(bins,observed=True).absolute_error.mean()
        if len(errors)>0:
            make_figure(folder/'error_by_activity.png',range(1,len(errors)+1),errors.values,
                        'Activity quantile','Mean absolute error','Held-out error by activity quintile',kind='bar')
            files.append('error_by_activity.png')
    else:
        threshold_y=first.observed.astype(int)
        if first.probability_active.notna().all() and len(set(threshold_y))==2:
            fpr,tpr,_=roc_curve(threshold_y,first.probability_active)
            make_figure(folder/'roc_curve.png',fpr,tpr,'False positive rate','True positive rate',
                        'Held-out ROC · first seed/model')
            files.append('roc_curve.png')
            precision,recall,_=precision_recall_curve(threshold_y,first.probability_active)
            make_figure(folder/'pr_curve.png',recall,precision,'Recall','Precision',
                        'Held-out precision–recall · first seed/model')
            files.append('pr_curve.png')
        cm=confusion_matrix(threshold_y,first.predicted.astype(int),labels=[0,1])
        fig,ax=plt.subplots(figsize=(5.2,4.7))
        fig.patch.set_facecolor(BG);ax.set_facecolor(BG)
        ax.imshow(cm,cmap='YlGnBu')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='black' if cm[i,j] else FG,fontsize=18)
        ax.set_xticks([0,1],['Inactive','Active']);ax.set_yticks([0,1],['Inactive','Active'])
        ax.set_xlabel('Predicted label',color=FG);ax.set_ylabel('Observed label',color=FG)
        ax.tick_params(colors=FG);ax.set_title('Held-out confusion matrix · first seed/model',color=FG)
        fig.tight_layout();fig.savefig(folder/'confusion_matrix.png',dpi=160,facecolor=BG);plt.close(fig)
        files.append('confusion_matrix.png')
    return files
