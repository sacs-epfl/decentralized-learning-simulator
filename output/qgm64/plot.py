"""Plot audited compact QGM results; no access to raw trained models needed."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot(root):
    root=Path(root)
    curves=pd.read_csv(root/'curves.csv')
    latest=pd.read_csv(root/'latest-aggregate.csv')
    names={'qgm':'QG-DSGDm (Nesterov)','dsgdm':'Local-momentum DSGD (Nesterov)',
           'dsgd':'Plain DSGD','conflux-v2':'Conflux V2: all nodes'}
    colors={'qgm':'#0072B2','dsgdm':'#D55E00','dsgd':'#009E73','conflux-v2':'#CC79A7'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.dpi':180})
    for metric,ylabel,x_col,scale,stem in [
            ('accuracy','Validation accuracy (%)','time',1,'accuracy'),
            ('loss','Validation cross-entropy','time',1,'loss'),
            ('accuracy','Validation accuracy (%)','bytes_sent',1e9,'traffic-accuracy')]:
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
        for i,profile in enumerate(['uniform','heterogeneous']):
            for j,part in enumerate(['iid','dirichlet']):
                ax=axes[i,j]
                for variant in names:
                    data=curves[(curves.profile==profile)&(curves.partitioner==part)&(curves.variant==variant)]
                    group=data.groupby('time')[metric].agg(['mean','std'])
                    x=data.groupby('time')[x_col].mean()/scale
                    ax.plot(x,group['mean'],color=colors[variant],label=names[variant],marker='o',markersize=3)
                    ax.fill_between(x,group['mean']-group['std'],group['mean']+group['std'],color=colors[variant],alpha=.12)
                ax.set_title(f'{profile.capitalize()} capacities / '+('IID' if part=='iid' else 'Dirichlet α=0.1'))
                ax.grid(alpha=.2)
                if x_col=='time':ax.set_xlim(150,900)
                if i==1:ax.set_xlabel('Simulated protocol time (s)' if x_col=='time' else 'Aggregate traffic (decimal GB)')
                if j==0:ax.set_ylabel(ylabel)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.015))
        fig.suptitle('CIFAR-10 / GN-LeNet / 64 nodes / one local step per exchange',y=.99)
        fig.text(.5,.002,'Fixed 1,000-image validation subset; mean ± seed SD (3 seeds). All-node stored-model means.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.1,1,.96))
        for ext in ['png','pdf']:fig.savefig(root/f'{stem}.{ext}',bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for i,profile in enumerate(['uniform','heterogeneous']):
        for j,part in enumerate(['iid','dirichlet']):
            ax=axes[i,j]
            for frame,label,style in [(curves[curves.variant=='conflux-v2'],'All stored node models','-'),
                                      (latest,'Latest completed sample aggregate','--')]:
                data=frame[(frame.profile==profile)&(frame.partitioner==part)]
                group=data.groupby('time').accuracy.agg(['mean','std'])
                ax.plot(group.index,group['mean'],style,color=colors['conflux-v2'],label=label,marker='o',markersize=3)
                ax.fill_between(group.index,group['mean']-group['std'],group['mean']+group['std'],color=colors['conflux-v2'],alpha=.08)
            ax.set_title(f'{profile.capitalize()} / '+('IID' if part=='iid' else 'Dirichlet α=0.1'))
            ax.grid(alpha=.2)
            if i==1:ax.set_xlabel('Simulated protocol time (s)')
            if j==0:ax.set_ylabel('Validation accuracy (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Conflux V2: distinct stored-model and completed-aggregate metrics')
    fig.tight_layout(rect=(0,.06,1,.96))
    for ext in ['png','pdf']:fig.savefig(root/f'v2-aggregate.{ext}',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args();plot(args.root)
