"""Generate docs/architecture.png — clean two-column flow diagram.

Left:  query pipeline (User → FastAPI → embed → pgvector → generate → response)
Right: eval pipeline (eval_set → run_all → metric modules → consolidated_report)

Muted professional palette matching the rest of the project. Run:
    python docs/_diagram.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Palette — muted, professional, matching career-ops CV gradient
TEAL = "#1e7a8c"
PURPLE = "#7a3fa5"
DARK = "#1a1a2e"
GRAY = "#888"
LIGHT_FILL = "#f7f8fb"
LIGHT_TEAL_FILL = "#eaf3f5"
LIGHT_PURPLE_FILL = "#f3edf7"
ACCENT_USER = "#cd6155"
ACCENT_USER_FILL = "#fdf2f0"
ACCENT_SUCCESS = "#229954"
ACCENT_SUCCESS_FILL = "#f0f8f4"


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=160)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(7, 8.0, "FamilyHQ RAG — Architecture", ha="center", fontsize=20,
            fontweight="bold", color=DARK, family="sans-serif")
    ax.text(7, 7.55, "Production query pipeline (left) + reproducible eval suite (right)",
            ha="center", fontsize=11, color=GRAY)

    def box(x, y, w, h, text, *, fill=LIGHT_FILL, edge=TEAL, fontsize=10,
            fontweight="normal", text_color=DARK):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            linewidth=1.6, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=fontweight,
                family="sans-serif")

    def arrow(x1, y1, x2, y2, *, color=TEAL, lw=1.8):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=2, shrinkB=2),
        )

    # ── Column geometry
    box_w = 5.6
    box_h = 0.55
    left_x = 0.4
    right_x = 8.0
    arrow_dx = 0.45  # arrow length between boxes
    pitch = 1.0       # vertical distance from box to box

    # ===== LEFT COLUMN: query pipeline =====
    ax.text(left_x + box_w / 2, 6.9, "QUERY PIPELINE",
            ha="center", fontsize=12, fontweight="bold", color=TEAL)

    y = 6.3
    box(left_x, y, box_w, box_h, "User query", fill=ACCENT_USER_FILL, edge=ACCENT_USER,
        fontweight="bold", text_color=ACCENT_USER)
    arrow(left_x + box_w / 2, y, left_x + box_w / 2, y - arrow_dx)

    y -= pitch
    box(left_x, y, box_w, box_h,
        "FastAPI  ·  POST /query  (ConnectionPool + shared Mistral client)",
        fontweight="bold")
    arrow(left_x + box_w / 2, y, left_x + box_w / 2, y - arrow_dx)

    y -= pitch
    box(left_x, y, box_w, box_h,
        "1.  Embed query  ·  Mistral mistral-embed  (1024-dim)")
    arrow(left_x + box_w / 2, y, left_x + box_w / 2, y - arrow_dx)

    y -= pitch
    box(left_x, y, box_w, box_h,
        "2.  pgvector HNSW search  ·  cosine top-k = 5",
        fill=LIGHT_TEAL_FILL)
    arrow(left_x + box_w / 2, y, left_x + box_w / 2, y - arrow_dx)

    y -= pitch
    box(left_x, y, box_w, box_h,
        "3.  Generate  ·  Mistral chat  (small  |  large)")
    arrow(left_x + box_w / 2, y, left_x + box_w / 2, y - arrow_dx)

    y -= pitch
    box(left_x, y, box_w, box_h, "Cited response  →  user",
        fill=ACCENT_SUCCESS_FILL, edge=ACCENT_SUCCESS,
        fontweight="bold", text_color=ACCENT_SUCCESS)

    # ===== RIGHT COLUMN: eval pipeline =====
    ax.text(right_x + box_w / 2, 6.9, "EVAL PIPELINE",
            ha="center", fontsize=12, fontweight="bold", color=PURPLE)

    y = 6.3
    box(right_x, y, box_w, box_h,
        "eval_set.jsonl  ·  30 hand-labeled queries",
        edge=PURPLE, fontweight="bold")
    arrow(right_x + box_w / 2, y, right_x + box_w / 2, y - arrow_dx, color=PURPLE)

    y -= pitch
    box(right_x, y, box_w, box_h,
        "run_all.py  ·  60 RAG calls  ·  checkpoint + retry",
        edge=PURPLE, fontweight="bold")
    arrow(right_x + box_w / 2, y, right_x + box_w / 2, y - arrow_dx, color=PURPLE)

    y -= pitch
    box(right_x, y, box_w, box_h,
        "retrieval  ·  latency  ·  cost  ·  judge  ·  refusal",
        fill=LIGHT_PURPLE_FILL, edge=PURPLE)
    arrow(right_x + box_w / 2, y, right_x + box_w / 2, y - arrow_dx, color=PURPLE)

    y -= pitch
    box(right_x, y, box_w, box_h,
        "consolidated_report.json  ·  per-query + aggregate",
        edge=PURPLE)
    arrow(right_x + box_w / 2, y, right_x + box_w / 2, y - arrow_dx, color=PURPLE)

    y -= pitch
    box(right_x, y, box_w, box_h,
        "README baseline table  ·  reproducible numbers",
        fill=LIGHT_PURPLE_FILL, edge=PURPLE, fontweight="bold")

    # ===== Footer =====
    ax.text(
        7, 0.25,
        "Stack:  Python 3.12  ·  Mistral API  ·  pgvector / Postgres 16 HNSW (m=16, ef_construction=64)  ·  FastAPI  ·  psycopg-pool  ·  MIT",
        ha="center", fontsize=9, color=GRAY,
    )

    out = "docs/architecture.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
