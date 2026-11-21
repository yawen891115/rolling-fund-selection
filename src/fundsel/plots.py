"""出图（英文标签：任何电脑都不需要中文字体就能正常显示）

配色：固定的分类色板，同一个策略在所有图里颜色都一样；基准/对照组用中性灰黑色。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # name -> (color, linestyle, linewidth)
    "PCA-6 + RP": ("#2a78d6", "-", 1.8),
    "AE-6 + RP": ("#eb6834", "-", 1.8),
    "Raw-30 + RP": ("#1baf7a", "-", 1.3),
    "Sharpe-12m + RP": ("#eda100", "-", 1.3),
    "PCA-6 + EW": ("#6250d6", "-", 1.3),
    "Universe EW": ("#8a8984", "--", 1.2),
    "CSI Equity Fund Index": (INK, "-", 1.4),
    "Equal-weight return proxy": (INK, "-", 1.4),
    "CSI 300 TR": ("#b5b4ae", ":", 1.4),
}


def _style_ax(ax, title: str, ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def _fig(w=9, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _spread(values: list[float], min_gap: float) -> list[float]:
    """把末端标签的纵坐标错开，保证相邻标签至少相隔 min_gap（避免数字叠在一起）"""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out.tolist()


def nav_curves(navs: dict[str, pd.Series], path: Path, title: str, shade_from: str | None = None):
    fig, ax = _fig()
    if shade_from is not None:   # mark the post-period hold-out
        any_s = next(iter(navs.values()))
        ax.axvspan(pd.Timestamp(shade_from), any_s.index[-1], color="#f1f0ec", zorder=0)
        ax.text(pd.Timestamp(shade_from), 0.99, "  hold-out", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color=INK2)
    ends = []
    for name, s in navs.items():
        c, ls, lw = STYLE.get(name, ("#999999", "-", 1.2))
        ax.plot(s.index, s.values, color=c, ls=ls, lw=lw, label=name, solid_capstyle="round")
        ax.plot(s.index[-1], s.values[-1], "o", ms=4.5, color=c, mec=SURFACE, mew=1.2)
        ends.append((s.index[-1], float(s.values[-1])))
    ax.axhline(1.0, color=GRID, lw=0.8)
    lo, hi = ax.get_ylim()
    ys = _spread([e[1] for e in ends], (hi - lo) * 0.035)
    for (x, y), y_lab in zip(ends, ys):
        ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(x + pd.Timedelta(days=12), y_lab),
                    textcoords="data", va="center", fontsize=7.5, color=INK2, annotation_clip=False)
    _style_ax(ax, title, "Net value (start = 1.0)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", labelcolor=INK2)
    ax.margins(x=0.07)
    _save(fig, path)


def drawdowns(rets: dict[str, pd.Series], path: Path, title: str):
    fig, ax = _fig(9, 3.6)
    for name, r in rets.items():
        nav = (1 + r).cumprod()
        dd = nav / np.maximum.accumulate(np.concatenate([[1.0], nav.values]))[1:] - 1
        c, ls, lw = STYLE.get(name, ("#999999", "-", 1.2))
        ax.plot(r.index, dd * 100, color=c, ls=ls, lw=lw, label=name)
    _style_ax(ax, title, "Drawdown (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
    _save(fig, path)


def yearly_bars(yearly: pd.DataFrame, path: Path, title: str):
    """yearly: index = 年份, columns = 策略名"""
    fig, ax = _fig(9, 4)
    n = len(yearly.columns)
    width = min(0.8 / n, 0.18)
    x = np.arange(len(yearly.index))
    for i, name in enumerate(yearly.columns):
        c = STYLE.get(name, ("#999999",))[0]
        vals = yearly[name].values * 100
        ax.bar(x + (i - (n - 1) / 2) * (width + 0.02), vals, width=width, color=c, label=name,
               edgecolor=SURFACE, linewidth=0.8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [str(y) for y in yearly.index])
    _style_ax(ax, title, "Calendar-year return (%)")
    ax.legend(frameon=False, fontsize=8, ncol=min(n, 4), loc="upper right", labelcolor=INK2)
    _save(fig, path)


def cumulative_ic(ics: dict[str, pd.Series], path: Path, title: str):
    fig, ax = _fig(9, 3.8)
    for name, ic in ics.items():
        c, ls, lw = STYLE.get(name, ("#999999", "-", 1.2))
        s = ic.sort_index().cumsum()
        ax.plot(s.index, s.values, color=c, ls=ls, lw=lw, label=f"{name}  (mean {ic.mean():+.3f})")
    ax.axhline(0, color=INK2, lw=0.8)
    _style_ax(ax, title, "Cumulative monthly Rank IC")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    _save(fig, path)


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("bgr", ["#184f95", "#6da7ec", "#f0efec", "#ec8a8a", "#a61f1f"])


def heatmap(mat: pd.DataFrame, path: Path, title: str, vmax: float = 1.0, fmt: str | None = None,
            figsize=(7.5, 8.5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(mat.values, cmap=_diverging_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=90 if mat.shape[1] > 8 else 0, fontsize=7)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7)
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if fmt:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 * vmax else INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    _save(fig, path)


def explained_variance(evr: list[float], path: Path, title: str):
    fig, ax = _fig(6.5, 3.4)
    k = np.arange(1, len(evr) + 1)
    ax.bar(k, np.array(evr) * 100, width=0.5, color="#2a78d6", label="Per component")
    ax.plot(k, np.cumsum(evr) * 100, "-o", color=INK, lw=1.4, ms=4, mec=SURFACE, label="Cumulative")
    for i, v in enumerate(np.cumsum(evr) * 100):
        ax.annotate(f"{v:.0f}%", (k[i], v), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(k, [f"PC{i}" for i in k])
    ax.set_ylim(0, 105)
    _style_ax(ax, title, "Variance explained (%)")
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=INK2)
    _save(fig, path)
