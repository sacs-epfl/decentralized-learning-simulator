"""Audit and summarize a completed, paired CIFAR baseline sweep."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def report(root):
    summaries, curves, traffic = [], [], []
    manifests, audits, fresh = [], [], []
    for profile in ["uniform", "heterogeneous"]:
        folder = root/profile
        if not (folder/"COMPLETE").exists():
            raise RuntimeError("Incomplete sweep: " + str(folder))
        manifest = json.loads((folder/"manifest.json").read_text())
        manifests.append(manifest)
        summary = pd.read_csv(folder/"summary.csv")
        curve = pd.read_csv(folder/"curves.csv")
        assert len(summary) == len(manifest["variants"])*len(manifest["partitioners"])*len(manifest["seeds"])
        assert not summary.duplicated(["variant","profile","partitioner","seed"]).any()
        for row in summary.to_dict("records"):
            key = {k:row[k] for k in ["variant","profile","partitioner","seed"]}
            run = folder/"runs"/f'{row["variant"]}-{profile}-{row["partitioner"]}-s{row["seed"]}'
            nodes = pd.read_csv(run/"accuracies.csv")
            counts = nodes.groupby("time").peer.nunique()
            expected_times = np.arange(manifest["test_period"],manifest["duration"]+.001,manifest["test_period"])
            assert np.array_equal(counts.index.to_numpy(),expected_times)
            assert counts.eq(manifest["peers"]).all() and len(nodes)==len(counts)*manifest["peers"]
            assert not nodes.duplicated(["peer","time"]).any()
            assert np.isfinite(nodes[["accuracy","loss"]]).all().all()
            assert nodes.accuracy.between(0,100).all()
            state = json.loads((run/"settings.json").read_text())
            assert state["model_size_bytes"]==manifest["model_bytes"]
            assert state["learning"]["momentum"]==0
            network = pd.read_csv(run/"network-snapshots.csv")
            assert np.array_equal(network.time,expected_times)
            assert network.bytes_sent.is_monotonic_increasing
            assert int(network.bytes_sent.iloc[-1])==int(row["bytes_sent"])
            for name,value in key.items():
                network[name]=value
            traffic.append(network)
            if row["variant"]=="conflux-v2":
                latest = pd.read_csv(run/"latest-aggregate.csv")
                assert len(latest)==len(counts)
                for name,value in key.items():latest[name]=value
                fresh.append(latest)
            audits.append({**key,"per_node_rows":len(nodes),"all_nodes_per_checkpoint":True,
                           "finite_metrics":True,"final_traffic_exact":True})
        summaries.append(summary);curves.append(curve)
    excluded={"profiles","output","cuda_visible_devices"}
    for key in manifests[0].keys()-excluded:
        assert manifests[0][key]==manifests[1][key],key
    summary=pd.concat(summaries,ignore_index=True);curve=pd.concat(curves,ignore_index=True)
    network=pd.concat(traffic,ignore_index=True)
    joined=curve.merge(network,on=["variant","profile","partitioner","seed","time"],validate="one_to_one")
    summary.to_csv(root/"summary.csv",index=False)
    joined.to_csv(root/"curves.csv",index=False)
    pd.concat(fresh,ignore_index=True).to_csv(root/"latest-v2-aggregate.csv",index=False)
    grouped=summary.groupby(["profile","partitioner","variant"])
    table=grouped.agg(seeds=("seed","nunique"),accuracy_mean=("final_accuracy","mean"),
        accuracy_sd=("final_accuracy","std"),loss_mean=("final_loss","mean"),
        loss_sd=("final_loss","std"),bytes_mean=("bytes_sent","mean"),
        training_tasks_mean=("training_tasks","mean"),execution_s_mean=("actual_execution_s","mean")).reset_index()
    table.to_csv(root/"comparison.csv",index=False)
    targets=[]
    for keys,g in joined.groupby(["profile","partitioner","variant","seed"]):
        for target in [20,30,40,50,60]:
            reached=g[g.accuracy>=target].sort_values("time")
            targets.append(dict(zip(["profile","partitioner","variant","seed"],keys)) | {
                "target_accuracy":target,"reached":bool(len(reached)),
                "time_s":None if reached.empty else float(reached.time.iloc[0]),
                "bytes_at_checkpoint":None if reached.empty else int(reached.bytes_sent.iloc[0]),
                "censor_time_s":manifests[0]["duration"]})
    pd.DataFrame(targets).to_csv(root/"time-to-accuracy.csv",index=False)
    (root/"audit.json").write_text(json.dumps({"passed":True,"runs":audits,
        "source_sha256":manifests[0]["source_sha256"],
        "readout":"Mean accuracy/loss across all physical nodes' stored post-aggregation checkpoints. Additional V2 latest-aggregate readout is distinct; no uncharged global parameter average.",
        "limitations":["Three training/capacity seeds; short CIFAR screening, not a universal ranking.",
            "Common hyperparameters, not individually tuned paper reproductions.",
            "Stable membership and oracle discovery; full published systems are not reproduced.",
            "Accuracy targets are first observed fixed checkpoints; unreached targets are censored.",
            "GPU wall time is recorded separately from simulated protocol time."]},indent=2)+"\n")
    plot(joined,root,manifests[0]["peers"],len(manifests[0]["seeds"]))
    print(table.to_string(index=False))


def plot(curves,root,peers,seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants=list(curves.variant.unique())
    colors=dict(zip(variants,plt.get_cmap("tab10").colors))
    for metric,xfield,name in [("accuracy","time","accuracy"),("loss","time","loss"),
                                ("accuracy","bytes_sent","traffic-accuracy")]:
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=False,sharey=True)
        for ax,((profile,partitioner),data) in zip(axes.flat,curves.groupby(["profile","partitioner"],sort=True)):
            for variant,g in data.groupby("variant",sort=False):
                stats=g.groupby("time").agg(mean=(metric,"mean"),sd=(metric,"std"),x=(xfield,"mean"))
                x=stats.x.to_numpy()/(1e9 if xfield=="bytes_sent" else 1)
                ax.plot(x,stats["mean"],color=colors[variant],label=variant)
                ax.fill_between(x,stats["mean"]-stats.sd,stats["mean"]+stats.sd,color=colors[variant],alpha=.12)
            ax.set(title=f"{profile}, {partitioner}",xlabel="Simulated protocol time [s]" if xfield=="time" else "Total traffic [GB]",
                   ylabel="All-node mean accuracy [%]" if metric=="accuracy" else "All-node mean cross-entropy")
            ax.grid(alpha=.25)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9)
        fig.suptitle(f"CIFAR-10 / GNLeNet, {peers} nodes; mean ± sample SD over {seeds} seeds")
        fig.tight_layout(rect=(0,.08,1,.96))
        fig.savefig(root/f"{name}.png",dpi=180);fig.savefig(root/f"{name}.svg")
        plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    report(p.parse_args().root)
