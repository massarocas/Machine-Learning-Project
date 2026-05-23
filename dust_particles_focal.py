"""
Focal loss vs BCE — logistic regression on synthetic imbalanced data.

Domain: optical particle counter — dust (majority, 0) vs core grit (minority, 1).

Three imbalance levels show that as the minority shrinks, BCE tends to ignore
core particles while focal loss keeps detecting them (higher balanced accuracy
and minority recall).
"""

from __future__ import annotations

import autograd.numpy as np
from autograd import grad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

def focal_loss(y_true, y_pred, gamma, alpha, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    p_t = np.where(y_true == 1, y_pred, 1.0 - y_pred)
    alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
    return np.mean(-alpha_t * (1.0 - p_t) ** gamma * np.log(p_t))


def binary_crossentropy(y_true, y_pred, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

class LogisticRegression:
    def __init__(self, lr=0.05, max_iters=400, tolerance=1e-5, penalty=None, C=0.01):
        self.lr, self.max_iters, self.tolerance = lr, max_iters, tolerance
        self.penalty, self.C = penalty, C
        self.theta, self.errors = None, []

    @staticmethod
    def sigmoid(x):
        return 0.5 * (np.tanh(0.5 * x) + 1)

    @staticmethod
    def _add_intercept(X):
        return np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)

    def _add_penalty(self, loss, w):
        if self.penalty == "l1":
            loss += self.C * np.abs(w[1:]).sum()
        elif self.penalty == "l2":
            loss += 0.5 * self.C * (w[1:] ** 2).sum()
        return loss

    def _loss(self, w):
        preds = self.sigmoid(np.dot(self.X, w))
        return self._add_penalty(binary_crossentropy(self.y, preds), w)

    def _cost(self, theta):
        return binary_crossentropy(self.y, self.sigmoid(self.X.dot(theta)))

    def fit(self, X, y):
        self.X = self._add_intercept(np.array(X, dtype=float))
        self.y = np.array(y, dtype=float)
        self.theta = np.random.normal(size=self.X.shape[1], scale=0.5)
        cost_d = grad(self._loss)
        self.errors = [self._cost(self.theta)]
        for _ in range(self.max_iters):
            self.theta -= self.lr * cost_d(self.theta)
            self.errors.append(self._cost(self.theta))
            if abs(self.errors[-2] - self.errors[-1]) < self.tolerance:
                break

    def predict_proba(self, X):
        X = self._add_intercept(np.array(X, dtype=float))
        return self.sigmoid(X.dot(self.theta))


class FocalLogisticRegression(LogisticRegression):
    def __init__(self, gamma=2.0, alpha=0.75, **kwargs):
        super().__init__(**kwargs)
        self.gamma, self.alpha = gamma, alpha

    def _loss(self, w):
        preds = self.sigmoid(np.dot(self.X, w))
        return self._add_penalty(
            focal_loss(self.y, preds, self.gamma, self.alpha), w
        )

    def _cost(self, theta):
        return focal_loss(
            self.y,
            self.sigmoid(self.X.dot(theta)),
            self.gamma,
            self.alpha,
        )
    
def make_data(n_dust, n_core, n_feat=4, sep=1.2):
    """Synthetic sensor features: dust (0) vs core grit (1)."""
    X_dust = np.random.randn(n_dust, n_feat)
    X_core = np.random.randn(n_core, n_feat) + sep
    X = np.vstack([X_dust, X_core])
    y = np.hstack([np.zeros(n_dust), np.ones(n_core)])
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

def split(X, y, test=0.25):
    n = int(len(y) * (1 - test))
    return X[:n], X[n:], y[:n], y[n:]

def metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    bal = (rec + spec) / 2
    acc = (tp + tn) / len(y_true)
    return dict(
        acc=acc,
        bal_acc=bal,
        precision=prec,
        recall=rec,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


SCENARIOS = [
    dict(name="Mild\n(80/20)", n_dust=800, n_core=200, alpha=0.80, label="mild"),
    dict(name="Moderate\n(87/13)", n_dust=870, n_core=130, alpha=0.87, label="moderate"),
    dict(name="Extreme\n(96/4)", n_dust=960, n_core=40, alpha=0.96, label="extreme"),
]

PURPLE = "#7F77DD"
GRAY = "#9A9893"
RED = "#E24B4A"
GREEN = "#1D9E75"
BG = "#FAFAF8"
BG2 = "#F2F1EF"
TEXT = "#1A1A18"
LIGHT = "#888780"


def bar_group(ax, bce_val, fl_val, metric_name, pct=True):
    fmt = lambda v: f"{v:.0%}" if pct else f"{v:.3f}"
    bars = ax.bar([0, 1], [bce_val, fl_val], color=[GRAY, PURPLE], width=0.55, zorder=3)
    for b, v in zip(bars, [bce_val, fl_val]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            fmt(v),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="500",
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BCE", "Focal"], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title(metric_name, fontsize=10, pad=4)
    ax.set_facecolor(BG2)
    ax.grid(axis="y", color="white", linewidth=1.2, zorder=2)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_yticklabels([])


def main():
    from pathlib import Path

    out_dir = Path(__file__).resolve().parent
    plot_path = out_dir / "focal_vs_bce.png"

    fig = plt.figure(figsize=(15, 12), facecolor=BG)
    fig.suptitle(
        "Focal loss vs BCE — dust vs core grit (synthetic imbalanced data)",
        fontsize=14,
        fontweight="600",
        color=TEXT,
        y=0.98,
    )
    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)
    all_results = []

    for row_idx, sc in enumerate(SCENARIOS):
        X, y = make_data(sc["n_dust"], sc["n_core"])
        Xtr, Xte, ytr, yte = split(X, y)

        bce_m = LogisticRegression(lr=0.05, max_iters=400)
        bce_m.fit(Xtr, ytr)
        bce_p = bce_m.predict_proba(Xte)

        fl_m = FocalLogisticRegression(
            gamma=2.0, alpha=sc["alpha"], lr=0.05, max_iters=400
        )
        fl_m.fit(Xtr, ytr)
        fl_p = fl_m.predict_proba(Xte)

        bm = metrics(yte, bce_p)
        fm = metrics(yte, fl_p)
        all_results.append((sc, bm, fm))

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 6, subplot_spec=outer[row_idx], wspace=0.35
        )

        ax_lbl = fig.add_subplot(inner[0])
        pct_core = sc["n_core"] / (sc["n_dust"] + sc["n_core"])
        ax_lbl.text(
            0.5,
            0.65,
            sc["name"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="600",
            color=TEXT,
            transform=ax_lbl.transAxes,
            multialignment="center",
        )
        ax_lbl.text(
            0.5,
            0.35,
            f"{sc['n_dust'] + sc['n_core']} particles\n"
            f"core grit: {pct_core:.0%}",
            ha="center",
            va="center",
            fontsize=8,
            color=LIGHT,
            transform=ax_lbl.transAxes,
            multialignment="center",
        )
        ax_lbl.axis("off")

        for col, (bv, fv, title) in enumerate(
            [
                (bm["acc"], fm["acc"], "Accuracy"),
                (bm["bal_acc"], fm["bal_acc"], "Balanced Acc ★"),
                (bm["recall"], fm["recall"], "Recall\n(core grit)"),
                (bm["f1"], fm["f1"], "F1\n(core grit)"),
            ]
        ):
            ax = fig.add_subplot(inner[col + 1])
            bar_group(ax, bv, fv, title)

        ax_cm = fig.add_subplot(inner[5])
        ax_cm.set_facecolor(BG2)
        ax_cm.axis("off")
        ax_cm.set_title("Confusion Δ\n(Focal − BCE)", fontsize=10, pad=4)

        for lbl, dv, good, x, y_pos in [
            ("TP", fm["tp"] - bm["tp"], True, 0.72, 0.30),
            ("FP", fm["fp"] - bm["fp"], False, 0.72, 0.70),
            ("FN", fm["fn"] - bm["fn"], False, 0.28, 0.30),
            ("TN", fm["tn"] - bm["tn"], True, 0.28, 0.70),
        ]:
            color = LIGHT
            if dv != 0:
                color = GREEN if (dv > 0) == good else RED
            ax_cm.text(
                x,
                y_pos,
                f"{lbl}\n{'+' + str(dv) if dv > 0 else dv}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="600",
                color=color,
                transform=ax_cm.transAxes,
            )

        ax_cm.text(
            0.5,
            0.04,
            "green=better  red=worse",
            ha="center",
            va="bottom",
            fontsize=7,
            color=LIGHT,
            transform=ax_cm.transAxes,
        )

    fig.text(
        0.5,
        0.005,
        "★ Balanced accuracy averages recall per class — it reveals models that "
        "only predict the majority (dust).",
        ha="center",
        fontsize=9,
        color=LIGHT,
        style="italic",
    )

    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()

    print(f"\n{'=' * 62}")
    print(f"  {'Imbalance':<14} {'Metric':<18} {'BCE':>7}  {'Focal':>7}  {'Δ':>7}")
    print(f"{'=' * 62}")
    for sc, bm, fm in all_results:
        label = sc["label"]
        for key, name in [
            ("acc", "Accuracy"),
            ("bal_acc", "Balanced Acc"),
            ("recall", "Recall (core)"),
            ("f1", "F1 (core)"),
        ]:
            delta = float(fm[key] - bm[key])
            arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "~")
            print(
                f"  {label:<14} {name:<18} {float(bm[key]):>7.3f}  "
                f"{float(fm[key]):>7.3f}  {arrow}{abs(delta):.3f}"
            )
        print(f"  {'':<14} {'-' * 44}")

    print(f"\nPlot saved → {plot_path}")


if __name__ == "__main__":
    main()
