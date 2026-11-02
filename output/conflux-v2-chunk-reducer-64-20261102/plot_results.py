"""Regenerate the final heatmaps from cells.csv and moshpit.csv (no raw runs)."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();root=args.directory
    cells=pd.read_csv(root/'cells.csv')
    baseline=pd.read_csv(root/'moshpit.csv')
    chunks=[8,16,32,64,128,256]
    reducers=[8,16,32,64]
    assert len(cells)==72 and cells.seeds.eq(3).all()
    assert len(baseline)==4 and baseline['count'].eq(3).all()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    for delta in [False,True]:
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        column='difference_mean_pp' if delta else 'accuracy_mean'
        limit=max(1,float(cells[column].abs().max()))
        norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit) if delta else None
        for row,partitioner in enumerate(['iid','dirichlet']):
            for col,profile in enumerate(['uniform','heterogeneous']):
                ax=axes[row,col]
                group=cells[(cells.profile==profile)&(cells.partitioner==partitioner)]
                values=group.pivot(index='reducers',columns='chunks',values=column).reindex(index=reducers,columns=chunks)
                im=ax.imshow(np.ma.masked_invalid(values.to_numpy()),origin='lower',aspect='auto',
                    cmap='RdBu_r' if delta else 'viridis',norm=norm,
                    **({} if delta else dict(vmin=float(cells.accuracy_mean.min()),vmax=float(cells.accuracy_mean.max()))))
                for y,rn in enumerate(reducers):
                    for x,cn in enumerate(chunks):
                        if rn>cn:
                            ax.add_patch(Rectangle((x-.5,y-.5),1,1,facecolor='#eeeeee',edgecolor='#cccccc',hatch='///',linewidth=.5))
                            ax.text(x,y,'—',ha='center',va='center',color='#555555',fontsize=16)
                            continue
                        c=group[(group.reducers==rn)&(group.chunks==cn)].iloc[0]
                        text=(f'{c.difference_mean_pp:+.2f}\n±{c.difference_sd_pp:.2f}' if delta
                              else f'{c.accuracy_mean:.2f}\n±{c.accuracy_sd:.2f}')
                        color=im.cmap(im.norm(float(values.loc[rn,cn])))
                        luminance=.2126*color[0]+.7152*color[1]+.0722*color[2]
                        ax.text(x,y,text,ha='center',va='center',fontsize=10,color='black' if luminance>.5 else 'white')
                b=baseline[(baseline.profile==profile)&(baseline.partitioner==partitioner)].iloc[0]
                ax.set(title=f'{profile.capitalize()} hardware · {"IID" if partitioner=="iid" else "Non-IID (α=0.1)"}\nMoshpit: {b["mean"]:.2f} ± {b["std"]:.2f}%',
                    xlabel='Number of chunks',ylabel='Reducer cap',xticks=range(len(chunks)),xticklabels=chunks,
                    yticks=range(len(reducers)),yticklabels=reducers)
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,
            label='Conflux V2 − Moshpit (percentage points)' if delta else 'Final all-node test accuracy (%)')
        fig.suptitle('Conflux V2 versus Moshpit at 900 simulated seconds' if delta else 'Conflux V2 accuracy at 900 simulated seconds',fontsize=17)
        fig.supxlabel('64 nodes · 16 V2 trainers · CIFAR-10 / GNLeNet · three paired seeds (mean ± sample SD)\n— / hatched: omitted because reducer cap > chunks; these cells are redundant, not failed runs.\nA reducer cap can exceed the number actually used on heterogeneous hardware.',fontsize=10)
        stem='moshpit-difference-heatmaps' if delta else 'accuracy-heatmaps'
        fig.savefig(root/f'{stem}.png',dpi=180)
        fig.savefig(root/f'{stem}.pdf')
        plt.close(fig)


if __name__=='__main__':main()
