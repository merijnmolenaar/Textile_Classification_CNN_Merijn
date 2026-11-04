"""Training curves for crop448_cv fold 0 (history.csv read from Drive)."""
import io, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = """epoch,train_loss,train_img_acc,val_loss,val_img_acc,val_smp_acc
1,0.3968,0.1824,0.1900,0.3544,0.3514
2,0.1575,0.4430,0.1295,0.5315,0.5135
3,0.1158,0.5933,0.1122,0.6366,0.6486
4,0.0967,0.6555,0.1041,0.6456,0.6577
5,0.0881,0.6799,0.1059,0.6486,0.6306
6,0.0757,0.7170,0.1094,0.6607,0.6757
7,0.0731,0.7184,0.1158,0.6547,0.7027
8,0.0621,0.7680,0.1121,0.6607,0.6847
9,0.0604,0.7680,0.0978,0.6877,0.7027
10,0.0496,0.8036,0.0984,0.6727,0.6757
11,0.0494,0.8218,0.1028,0.7357,0.7387
12,0.0449,0.8351,0.1026,0.7417,0.7658
13,0.0408,0.8470,0.0946,0.7297,0.7297
14,0.0332,0.8679,0.0986,0.7838,0.8108
15,0.0384,0.8588,0.0920,0.7628,0.7748
16,0.0305,0.8882,0.0945,0.7658,0.7748
17,0.0252,0.9043,0.0949,0.7297,0.7387
18,0.0230,0.9147,0.1113,0.7027,0.7207
19,0.0211,0.9210,0.1234,0.7658,0.7568
20,0.0201,0.9175,0.1038,0.7628,0.7477
21,0.0191,0.9280,0.1069,0.7628,0.7748
22,0.0193,0.9329,0.1086,0.7447,0.7477
23,0.0156,0.9392,0.1115,0.7658,0.7928
24,0.0190,0.9301,0.0971,0.7988,0.8018
"""
d = pd.read_csv(io.StringIO(CSV))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
best = int(d.loc[d.val_smp_acc.idxmax(), "epoch"])

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)


def style(ax, ylabel):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 27.5)
    ax.axvline(best, color=INK2, lw=1, ls=(0, (3, 3)))


def line(ax, y, c, label, mk, dy=0):
    ax.plot(d.epoch, y, color=c, lw=2, marker=mk, ms=4, label=label)
    ax.annotate(label, (d.epoch.iloc[-1], y.iloc[-1]), xytext=(6, dy),
                textcoords="offset points", va="center", color=INK, fontsize=9)

style(a1, "BCE loss")
line(a1, d.train_loss, BLUE, "train", "o")
line(a1, d.val_loss, ORANGE, "validation", "s")
a1.set_title("Loss", loc="left", color=INK, fontsize=11)

style(a2, "exact-match accuracy (%)")
line(a2, d.train_img_acc * 100, BLUE, "train (image)", "o")
line(a2, d.val_img_acc * 100, ORANGE, "val (image)", "s", dy=-8)
line(a2, d.val_smp_acc * 100, AQUA, "val (sample)", "^", dy=8)
a2.set_ylim(0, 100)
for ax in (a1, a2):
    ax.legend(loc="center right" if ax is a1 else "lower right", frameon=False,
              labelcolor=INK, fontsize=9)
a2.set_title("Accuracy", loc="left", color=INK, fontsize=11)
a2.annotate(f"selected epoch {best}\nval sample {d.val_smp_acc.max()*100:.1f}%",
            (best, 22), xytext=(-6, 0), textcoords="offset points", ha="right",
            color=INK2, fontsize=8.5)

fig.suptitle("crop448_cv, fold 0: training curves (validation = 111 samples; not the test score)",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=160, facecolor=SURF)
d.to_csv(sys.argv[1].replace(".png", ".csv"), index=False)
print("saved", sys.argv[1], "best epoch", best)
