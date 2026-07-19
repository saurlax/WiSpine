from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from wispine.csi import read_amplitude_phase, read_phase_ratio

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "papers" / "figures"
FILE_METRICS = ROOT / "outputs" / "file_phase_ratio_cloud_20260718" / "metrics.json"
EXTRATREES_MODEL = ROOT / "outputs" / "file_phase_ratio_cloud_20260718" / "model.joblib"

CLASS_ORDER = ("kyphosis", "normal", "scoliosis")
CLASS_LABELS = ("脊柱后凸", "正常姿态", "脊柱侧弯")
COLORS = ("#3B6FB6", "#3A9D76", "#D67A36")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 360,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    plt.close(fig)


def plot_csi_example() -> None:
    path = ROOT / "data" / "normal" / "cyd-n01.dat"
    metrics = read_amplitude_phase(path)
    ratio = read_phase_ratio(path)

    frame_limit = min(1024, metrics.frame_count)
    amplitude = metrics.amplitude[:frame_limit, :, 0, 0].T
    raw_ratio = ratio.phase_ratio[:frame_limit, 14, 0, 0]
    unwrapped_ratio = np.unwrap(raw_ratio)
    frames = np.arange(frame_limit)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.35),
        gridspec_kw={"height_ratios": (1.15, 1), "hspace": 0.58},
    )
    image = axes[0].imshow(
        amplitude,
        aspect="auto",
        origin="lower",
        extent=(0, frame_limit - 1, 1, amplitude.shape[0]),
        cmap="viridis",
    )
    axes[0].set_title("(a) CSI 幅度时-频分布", fontsize=10, pad=5)
    axes[0].set_xlabel("时间帧")
    axes[0].set_ylabel("子载波索引")
    axes[0].grid(False)
    colorbar = fig.colorbar(image, ax=axes[0], pad=0.015, aspect=28)
    colorbar.set_label("幅度")

    axes[1].plot(frames, raw_ratio, color="#A8A8A8", linewidth=0.75, label="原始相位比")
    axes[1].plot(frames, unwrapped_ratio, color=COLORS[0], linewidth=1.05, label="时间展开后")
    axes[1].set_title("(b) 相位比展开前后对比", fontsize=10, pad=5)
    axes[1].set_xlabel("时间帧")
    axes[1].set_ylabel("相位 / rad")
    axes[1].legend(
        fontsize=8,
        ncol=1,
        loc="lower left",
    )

    save(fig, "csi_signal_example")


def plot_dataset_composition() -> None:
    subjects = ("cyd", "dsy", "gxq", "ljy", "zzy")
    counts = np.zeros((len(CLASS_ORDER), len(subjects)), dtype=int)
    for class_index, class_name in enumerate(CLASS_ORDER):
        file_subjects = [
            path.stem.split("-")[0] for path in (ROOT / "data" / class_name).glob("*.dat")
        ]
        class_counts = Counter(file_subjects)
        counts[class_index] = [class_counts[subject] for subject in subjects]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(subjects))
    width = 0.23
    for index, (label, color) in enumerate(zip(CLASS_LABELS, COLORS, strict=True)):
        bars = ax.bar(x + (index - 1) * width, counts[index], width, label=label, color=color)
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(x, [name.upper() for name in subjects])
    ax.set_xlabel("受试者编号")
    ax.set_ylabel("原始 CSI 文件数")
    ax.set_ylim(0, max(60, counts.max() + 8))
    ax.legend(frameon=False, ncol=3, loc="upper center")
    save(fig, "dataset_composition")


def plot_dataset_split() -> None:
    data = json.loads(FILE_METRICS.read_text(encoding="utf-8"))
    split_names = ("train", "val", "test")
    split_labels = ("训练集", "验证集", "测试集")
    counts = np.array(
        [
            [
                Counter(item["class_name"] for item in data["split"][split])[name]
                for name in CLASS_ORDER
            ]
            for split in split_names
        ]
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(split_names))
    bottom = np.zeros(len(split_names), dtype=int)
    for class_index, (label, color) in enumerate(zip(CLASS_LABELS, COLORS, strict=True)):
        bars = ax.bar(
            x, counts[:, class_index], bottom=bottom, width=0.58, label=label, color=color
        )
        ax.bar_label(
            bars,
            labels=[str(value) for value in counts[:, class_index]],
            label_type="center",
            color="white",
            fontsize=9,
        )
        bottom += counts[:, class_index]
    ax.bar_label(
        ax.containers[-1], labels=[f"合计 {value}" for value in bottom], padding=4, fontsize=9
    )
    ax.set_xticks(x, split_labels)
    ax.set_ylabel("原始 CSI 文件数")
    ax.set_ylim(0, bottom.max() * 1.13)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    save(fig, "dataset_split")


def plot_extratrees_feature_importance() -> None:
    model = joblib.load(EXTRATREES_MODEL)
    importance = np.asarray(model.feature_importances_)
    feature_count = 180
    statistic_names = (
        "均值",
        "标准差",
        "10% 分位数",
        "中位数",
        "90% 分位数",
        "差分绝对值均值",
        "差分标准差",
        "正弦均值",
        "余弦均值",
    )
    grouped = importance.reshape(len(statistic_names), feature_count).sum(axis=1) * 100

    top_count = 10
    top_indices = np.argsort(importance)[-top_count:][::-1]
    detail_labels: list[str] = []
    for index in top_indices:
        statistic_index, signal_index = divmod(int(index), feature_count)
        subcarrier, remainder = divmod(signal_index, 6)
        compared_rx, tx = divmod(remainder, 3)
        detail_labels.append(
            f"{statistic_names[statistic_index]}\n"
            f"SC{subcarrier + 1}, RX{compared_rx + 2}/RX1, TX{tx + 1}"
        )

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": (0.86, 1.3)}, constrained_layout=True
    )
    order = np.argsort(grouped)
    axes[0].barh(np.arange(len(order)), grouped[order], color="#4F78B5")
    axes[0].set_yticks(np.arange(len(order)), np.asarray(statistic_names)[order])
    axes[0].set_xlabel("累计重要性 / %")
    axes[0].set_title("(a) 按统计量聚合")

    detail_values = importance[top_indices][::-1] * 100
    detail_labels = detail_labels[::-1]
    axes[1].barh(np.arange(top_count), detail_values, color="#D67A36")
    axes[1].set_yticks(np.arange(top_count), detail_labels, fontsize=7.4)
    axes[1].set_xlabel("特征重要性 / %")
    axes[1].set_title("(b) 重要性最高的 10 个具体特征")
    save(fig, "extratrees_feature_importance")


def plot_model_comparison() -> None:
    models = ("TCN", "RBF SVM", "Random Forest", "ExtraTrees")
    validation = np.array([56.12, 92.04, 96.46, 95.58])
    test = np.array([50.30, 89.38, 90.27, 92.04])
    x = np.arange(len(models))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    val_bars = ax.bar(x - width / 2, validation, width, color="#7895C5", label="验证准确率")
    test_bars = ax.bar(x + width / 2, test, width, color="#D67A36", label="测试准确率")
    ax.bar_label(val_bars, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(test_bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, models)
    ax.set_ylabel("准确率 / %")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "model_comparison")


def plot_confusion_matrix() -> None:
    data = json.loads(FILE_METRICS.read_text(encoding="utf-8"))
    matrix = np.asarray(data["test"]["confusion_matrix"])
    row_percent = matrix / matrix.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5.4, 4.35))
    image = ax.imshow(row_percent, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(3), CLASS_LABELS)
    ax.set_yticks(np.arange(3), CLASS_LABELS)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.grid(False)
    for row in range(3):
        for column in range(3):
            color = "white" if row_percent[row, column] > 55 else "#222222"
            ax.text(
                column,
                row,
                f"{matrix[row, column]}\n({row_percent[row, column]:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("按真实类别归一化 / %")
    save(fig, "extratrees_confusion_matrix")


def main() -> None:
    configure_style()
    plot_csi_example()
    plot_dataset_composition()
    plot_dataset_split()
    plot_extratrees_feature_importance()
    plot_model_comparison()
    plot_confusion_matrix()
    print(f"Generated 6 paper figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
