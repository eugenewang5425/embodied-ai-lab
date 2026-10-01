"""Round-14 paired outcomes, all stability cells, and fixed-seed real trajectories."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from embodied_learning.plotting import configure_plot_font

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "bypass_low": "14cm低路障",
    "bypass_beam": "42–56cm悬空杆",
    "bypass_crate": "1m普通箱体",
    "clear_overhead": "75cm高杆",
}
METHODS = {"corridor_depth": ("二维择路参照", "#247cc1"),
           "height_depth": ("三维部件择路", "#119c74")}


def small_arrays(folder, row):
    path = folder / row["file"]
    h = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(1 << 20), b""):
            h.update(part)
    assert h.hexdigest() == row["sha256"]
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("truth", "goal", "actual_boxes")}


def trajectory(ax, a, color, style="-"):
    p = a["truth"]
    ax.plot(p[:, 0], p[:, 1], color=color, ls=style, lw=2)


def scene(ax, a, overhead=False):
    for box in a["actual_boxes"][:-1]:
        ax.add_patch(Rectangle((box[0], box[2]), box[1]-box[0], box[3]-box[2], fc="#d8dee6"))
    box = a["actual_boxes"][-1]
    ax.add_patch(Rectangle((box[0], box[2]), box[1]-box[0], box[3]-box[2],
                           fc="none" if overhead else "#e8b06f", ec="#a76815",
                           ls="--" if overhead else "-", alpha=0.7))
    ax.plot(*a["goal"][-1], "*", color="#c38b00", ms=12)
    ax.set(xlim=(1, 11), ylim=(4.25, 7.8), aspect="equal", xlabel="x (m)", ylabel="y (m)")
    ax.tick_params(labelsize=9)


def main():
    configure_plot_font()
    folder = ROOT / "results/navigation_reinforcement_v14"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf8"))
    assert len(summary["rows"]) == 75
    rows = summary["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    x = np.arange(4)
    for i, (method, (label, color)) in enumerate(METHODS.items()):
        paired = [r for r in rows if r["method"] == method and r["layout"] == "street"
                  and r["seed"] in (37, 38, 39)]
        counts = [sum(r["passed"] for r in paired if r["profile"] == p) for p in PROFILES]
        bars = a.bar(x+(i-.5)*.32, counts, .3, color=color, label=label)
        a.bar_label(bars, labels=[f"{n}/3" for n in counts], padding=5, fontsize=10)
    a.set(xticks=x, xticklabels=list(PROFILES.values()), ylim=(0, 3.65), yticks=(0,1,2,3),
          title="同条件12对：到达、停稳且接触合格", ylabel="通过回合数")
    a.tick_params(axis="x", labelsize=9)
    a.grid(axis="y", alpha=.15)
    layouts = ("street", "street_west", "street_north")
    counts = np.array([[sum(r["passed"] for r in rows if r["method"] == "height_depth"
                           and r["layout"] == layout and r["profile"] == profile)
                        for profile in PROFILES] for layout in layouts])
    b.imshow(counts, cmap="Greens", vmin=0, vmax=5, aspect="auto")
    for (y, x0), n in np.ndenumerate(counts):
        b.text(x0, y, f"{n}/5", ha="center", va="center", color="white" if n>=3 else "black", fontsize=14)
    b.set(xticks=range(4), xticklabels=["低路障","悬空杆","箱体","高杆"],
          yticks=range(3), yticklabels=["原街区","西移宽障碍","北移障碍"],
          title="最终候选60回合：每格五个新种子")
    fig.legend(loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(Path(__file__).parent / "lesson69-round14-outcomes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.3), layout="constrained")
    for ax, profile in zip(axes.flat, PROFILES, strict=True):
        notes = []
        for method, (label, color) in METHODS.items():
            row = next(r for r in rows if r["method"]==method and r["layout"]=="street"
                       and r["seed"]==37 and r["profile"]==profile)
            arrays = small_arrays(folder, row)
            trajectory(ax, arrays, color, "--" if method=="corridor_depth" else "-")
            notes.append(f"{label}：{'到达' if row['passed'] else row['status']}，{row['elapsed_sim_s']:.1f}s")
        scene(ax, arrays, profile=="clear_overhead")
        ax.set_title(PROFILES[profile], fontsize=12)
        ax.text(.015, .02, "\n".join(notes), transform=ax.transAxes, fontsize=9,
                bbox={"facecolor":"white", "alpha":.9, "edgecolor":"none"})
    fig.legend([Line2D([],[],color="#247cc1",ls="--",lw=2),
                Line2D([],[],color="#119c74",lw=2),
                Line2D([],[],color="#c38b00",marker="*",ls="none")],
               ["二维择路参照","三维部件择路","真实目标"], loc="outside upper center", ncol=3,frameon=False)
    fig.savefig(Path(__file__).parent / "lesson69-round14-trajectories.png", dpi=170,bbox_inches="tight")
    plt.close(fig)

    old_folder = ROOT / "results/navigation_reinforcement_v13"
    dev_folder = ROOT / "results/navigation_height_development_v14a"
    old = json.loads((old_folder / "summary.json").read_text(encoding="utf8"))
    dev = json.loads((dev_folder / "summary.json").read_text(encoding="utf8"))
    original = next(r for r in old["rows"] if r["layout"]=="street" and r["profile"]=="bypass_beam"
                    and r["seed"]==33 and r["method"]=="corridor_depth")
    repaired = dev["rows"][0]
    assert repaired["profile"]=="bypass_beam" and repaired["seed"]==33
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for ax, directory, row, title, color in (
        (axes[0],old_folder,original,"已见种子33：二维压平，误拒退出","#247cc1"),
        (axes[1],dev_folder,repaired,"同一反例：保留高度后真实到达","#119c74"),
    ):
        arrays=small_arrays(directory,row)
        trajectory(ax,arrays,color)
        scene(ax,arrays)
        ax.plot(*arrays["truth"][-1,:2],"o" if row["passed"] else "x",color=color,ms=9)
        ax.set_title(title,fontsize=12)
        ax.text(.015,.02,f"峰值接触力 {row['contact_peak_force_n']:.2f}N；{row['elapsed_sim_s']:.1f}s",
                transform=ax.transAxes,fontsize=10,bbox={"facecolor":"white","alpha":.9,"edgecolor":"none"})
    fig.suptitle("开发机制检查：不能计入新种子留出成功率",fontsize=12)
    fig.savefig(Path(__file__).parent / "lesson69-round14-counterexample.png",dpi=170,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
