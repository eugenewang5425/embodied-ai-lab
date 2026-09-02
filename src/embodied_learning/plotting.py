"""Shared plot configuration for readable Chinese experiment reports."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_plot_font() -> None:
    """Use the Windows Microsoft YaHei font when it is available."""
    font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc"
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["axes.unicode_minus"] = False
