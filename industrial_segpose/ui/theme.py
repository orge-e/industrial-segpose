"""Deep Graphite design tokens for the Tk desktop workstation.

The theme intentionally stays inside stable ttk capabilities: color hierarchy,
typography, spacing and widget states.  It does not emulate web shadows,
gradients or animated controls with Canvas hacks.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


COLORS = {
    "app": "#0B1018",
    "sidebar": "#101722",
    "panel": "#151D29",
    "elevated": "#192332",
    "canvas": "#070B11",
    "border": "#273346",
    "border_soft": "#202B3A",
    "primary": "#3B82F6",
    "primary_hover": "#5795F8",
    "primary_pressed": "#2563EB",
    "text": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_disabled": "#596579",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "selection": "#1E3A5F",
}

FONT = {
    "family": "Microsoft YaHei UI",
    "app_title": 19,
    "page_title": 18,
    "section": 13,
    "body": 10,
    "secondary": 9,
    "metric": 22,
}

SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32}


def _font(size: int, weight: str = "normal") -> tuple[str, int, str]:
    return FONT["family"], size, weight


def configure_industrial_theme(root: tk.Misc) -> ttk.Style:
    """Configure the shared Deep Graphite ttk design system."""

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    root.configure(background=COLORS["app"])

    # Foundations and surfaces.
    style.configure("TFrame", background=COLORS["app"])
    style.configure("App.TFrame", background=COLORS["app"])
    style.configure("Header.TFrame", background=COLORS["sidebar"])
    style.configure("Sidebar.TFrame", background=COLORS["sidebar"])
    style.configure("Panel.TFrame", background=COLORS["panel"])
    style.configure("Elevated.TFrame", background=COLORS["elevated"])
    style.configure("Inspector.TFrame", background=COLORS["panel"])
    style.configure("Toolbar.TFrame", background=COLORS["panel"])
    style.configure("Footer.TFrame", background=COLORS["sidebar"])
    style.configure("Divider.TFrame", background=COLORS["border"])
    style.configure("Card.TFrame", background=COLORS["panel"])
    style.configure(
        "Card.TLabelframe",
        background=COLORS["panel"],
        bordercolor=COLORS["border_soft"],
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=COLORS["panel"],
        foreground=COLORS["text_secondary"],
        font=_font(FONT["secondary"], "bold"),
    )

    # Typography.
    style.configure("TLabel", background=COLORS["app"], foreground=COLORS["text"], font=_font(FONT["body"]))
    style.configure("HeaderTitle.TLabel", background=COLORS["sidebar"], foreground=COLORS["text"], font=_font(FONT["app_title"], "bold"))
    style.configure("HeaderSub.TLabel", background=COLORS["sidebar"], foreground=COLORS["text_secondary"], font=_font(FONT["secondary"]))
    style.configure("HeaderBadge.TLabel", background=COLORS["primary"], foreground="#FFFFFF", font=("Segoe UI", 10, "bold"), padding=(9, 6))
    style.configure("HeaderChip.TLabel", background=COLORS["elevated"], foreground=COLORS["text_secondary"], font=("Segoe UI", 9), padding=(10, 5))
    style.configure("PageTitle.TLabel", background=COLORS["app"], foreground=COLORS["text"], font=_font(FONT["page_title"], "bold"))
    style.configure("PageHint.TLabel", background=COLORS["app"], foreground=COLORS["text_secondary"], font=_font(FONT["body"]))
    style.configure("Title.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=_font(FONT["section"], "bold"))
    style.configure("Section.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=_font(FONT["section"], "bold"))
    style.configure("InspectorTitle.TLabel", background=COLORS["panel"], foreground=COLORS["text_secondary"], font=_font(FONT["secondary"], "bold"))
    style.configure("InspectorLabel.TLabel", background=COLORS["panel"], foreground=COLORS["text_secondary"], font=_font(FONT["secondary"]))
    style.configure("InspectorValue.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=_font(FONT["body"], "bold"))
    style.configure("Hint.TLabel", background=COLORS["panel"], foreground=COLORS["text_secondary"], font=_font(FONT["secondary"]))
    style.configure("StatusSuccess.TLabel", background=COLORS["panel"], foreground=COLORS["success"], font=_font(FONT["secondary"], "bold"))
    style.configure("StatusWarning.TLabel", background=COLORS["panel"], foreground=COLORS["warning"], font=_font(FONT["secondary"], "bold"))
    style.configure("StatusError.TLabel", background=COLORS["panel"], foreground=COLORS["error"], font=_font(FONT["secondary"], "bold"))

    # Semantic buttons.
    style.configure("TButton", background=COLORS["elevated"], foreground=COLORS["text"], font=_font(FONT["body"]), padding=(12, 8), borderwidth=0, focusthickness=0)
    style.map("TButton", background=[("active", COLORS["border"]), ("pressed", COLORS["selection"]), ("disabled", COLORS["panel"])], foreground=[("disabled", COLORS["text_disabled"])])
    style.configure("Primary.TButton", background=COLORS["primary"], foreground="#FFFFFF", font=_font(FONT["body"], "bold"), padding=(14, 9))
    style.map("Primary.TButton", background=[("active", COLORS["primary_hover"]), ("pressed", COLORS["primary_pressed"]), ("disabled", COLORS["border"])], foreground=[("disabled", COLORS["text_disabled"])])
    style.configure("Secondary.TButton", background=COLORS["elevated"], foreground=COLORS["text"], padding=(12, 8))
    style.map("Secondary.TButton", background=[("active", COLORS["border"]), ("pressed", COLORS["selection"]), ("disabled", COLORS["panel"])], foreground=[("disabled", COLORS["text_disabled"])])
    style.configure("Ghost.TButton", background=COLORS["panel"], foreground=COLORS["text_secondary"], padding=(10, 7))
    style.map("Ghost.TButton", background=[("active", COLORS["elevated"]), ("pressed", COLORS["selection"])], foreground=[("active", COLORS["text"]), ("disabled", COLORS["text_disabled"])])
    style.configure("Icon.TButton", background=COLORS["panel"], foreground=COLORS["text_secondary"], padding=(10, 6), font=("Segoe UI", 10))
    style.map("Icon.TButton", background=[("active", COLORS["elevated"]), ("pressed", COLORS["selection"])], foreground=[("active", COLORS["text"])])
    style.configure("Danger.TButton", background=COLORS["panel"], foreground=COLORS["error"], padding=(12, 8))
    style.map("Danger.TButton", background=[("active", "#321A20"), ("pressed", "#451C24")])

    # Left navigation.
    style.configure("Nav.TButton", background=COLORS["sidebar"], foreground=COLORS["text_secondary"], anchor="w", padding=(18, 11), font=_font(FONT["body"]), borderwidth=0)
    style.map("Nav.TButton", background=[("active", COLORS["elevated"])], foreground=[("active", COLORS["text"])])
    style.configure("NavActive.TButton", background=COLORS["selection"], foreground=COLORS["text"], anchor="w", padding=(18, 11), font=_font(FONT["body"], "bold"), borderwidth=0)
    style.map("NavActive.TButton", background=[("active", COLORS["selection"]), ("pressed", COLORS["selection"])])
    style.configure("NavSection.TLabel", background=COLORS["sidebar"], foreground=COLORS["text_disabled"], font=_font(FONT["secondary"], "bold"))

    # Authoring stepper.
    for state, background, foreground in (
        ("Idle", COLORS["sidebar"], COLORS["text_disabled"]),
        ("Active", COLORS["selection"], COLORS["text"]),
        ("Done", COLORS["sidebar"], COLORS["success"]),
        ("Error", COLORS["sidebar"], COLORS["error"]),
    ):
        style.configure(f"Stepper{state}.TFrame", background=background)
        style.configure(f"Stepper{state}.TLabel", background=background, foreground=foreground, font=_font(FONT["body"], "bold" if state == "Active" else "normal"))
        style.configure(f"Stepper{state}Index.TLabel", background=background, foreground=foreground, font=("Segoe UI", 10, "bold"))

    # Inputs and selection controls.
    style.configure("TEntry", fieldbackground=COLORS["elevated"], foreground=COLORS["text"], insertcolor=COLORS["text"], bordercolor=COLORS["border"], lightcolor=COLORS["border"], darkcolor=COLORS["border"], padding=7)
    style.map("TEntry", bordercolor=[("focus", COLORS["primary"]), ("disabled", COLORS["border_soft"])], foreground=[("disabled", COLORS["text_disabled"])])
    style.configure("TCombobox", fieldbackground=COLORS["elevated"], background=COLORS["elevated"], foreground=COLORS["text"], arrowcolor=COLORS["text_secondary"], bordercolor=COLORS["border"], padding=6)
    style.map("TCombobox", fieldbackground=[("readonly", COLORS["elevated"])], foreground=[("readonly", COLORS["text"])], bordercolor=[("focus", COLORS["primary"])])
    style.configure("TCheckbutton", background=COLORS["panel"], foreground=COLORS["text_secondary"], font=_font(FONT["secondary"]))
    style.map("TCheckbutton", background=[("active", COLORS["panel"])], foreground=[("active", COLORS["text"])])
    style.configure("TRadiobutton", background=COLORS["panel"], foreground=COLORS["text_secondary"])
    style.configure("TSeparator", background=COLORS["border_soft"])
    style.configure("TProgressbar", background=COLORS["primary"], troughcolor=COLORS["elevated"], borderwidth=0)

    # Notebook tabs are hidden; page selection is owned by the left navigation.
    style.configure("Workspace.TNotebook", background=COLORS["app"], borderwidth=0, tabmargins=0)
    try:
        style.layout("Workspace.TNotebook.Tab", [])
    except tk.TclError:
        pass

    # Shared production table treatment for subsequent phases.
    style.configure("Treeview", background=COLORS["panel"], fieldbackground=COLORS["panel"], foreground=COLORS["text"], rowheight=32, borderwidth=0, relief="flat", font=_font(FONT["secondary"]))
    style.configure("Treeview.Heading", background=COLORS["elevated"], foreground=COLORS["text_secondary"], relief="flat", padding=(7, 8), font=_font(FONT["secondary"], "bold"))
    style.map("Treeview", background=[("selected", COLORS["selection"])], foreground=[("selected", COLORS["text"])])
    style.map("Treeview.Heading", background=[("active", COLORS["border_soft"])])

    # Existing metric cards remain compatible on the untouched pages.
    style.configure("Metric.TFrame", background=COLORS["elevated"], relief="flat")
    style.configure("MetricValue.TLabel", background=COLORS["elevated"], foreground=COLORS["text"], font=_font(FONT["metric"], "bold"))
    style.configure("MetricLabel.TLabel", background=COLORS["elevated"], foreground=COLORS["text_secondary"], font=_font(FONT["secondary"]))
    style.configure("Footer.TLabel", background=COLORS["sidebar"], foreground=COLORS["text_secondary"], font=_font(FONT["secondary"]), padding=(12, 7))

    return style
