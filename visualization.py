"""Generate plots from recorded experiment output, never from synthetic placeholders."""
from __future__ import annotations
from pathlib import Path
import json
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def generate_artifacts(result_path: str):
    if not HAS_MPL:
        return "plots_skipped_no_matplotlib", None
    payload=json.loads(Path(result_path).read_text(encoding="utf-8")); rounds=payload.get("rounds",[])
    if not rounds: raise ValueError("No rounds recorded")
    out=Path(result_path).parent; stem=Path(result_path).stem
    ids=[r["round"] for r in rounds]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(ids,[r["global_accuracy"] for r in rounds],marker="o"); axes[0].set(title="Global accuracy",xlabel="Federated round",ylabel="Accuracy",ylim=(0,1)); axes[0].grid()
    axes[1].plot(ids,[r["global_loss"] for r in rounds],marker="o",color="tab:red"); axes[1].set(title="Global loss",xlabel="Federated round",ylabel="Rate loss"); axes[1].grid()
    fig.tight_layout(); metrics_path=out/f"{stem}_metrics.png"; fig.savefig(metrics_path,dpi=150); plt.close(fig)
    cm=rounds[-1].get("held_out_test",{}).get("confusion_matrix")
    if cm is not None:
        fig,ax=plt.subplots(figsize=(5,4)); image=ax.imshow(cm,cmap="Blues"); fig.colorbar(image,ax=ax); ax.set(title="Held-out synthetic test confusion matrix",xlabel="Predicted",ylabel="Actual")
        for i,row in enumerate(cm):
            for j,value in enumerate(row): ax.text(j,i,str(value),ha="center",va="center")
        fig.tight_layout(); cm_path=out/f"{stem}_confusion_matrix.png"; fig.savefig(cm_path,dpi=150); plt.close(fig)
    else: cm_path=None
    return str(metrics_path), str(cm_path) if cm_path else None
