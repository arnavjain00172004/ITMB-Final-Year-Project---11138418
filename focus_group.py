"""
Thematic Analysis of Focus Group Discussion
============================================
Participants: Student, Professional, Expert
Topic:        Perceptions of AI-based stock prediction tools
Format:       Six moderator-led discussion topics, each with one
              contribution from each of the three participants
              (18 contributions in total).

Method
------
Reflexive thematic analysis following the six-phase approach of
Braun and Clarke (2006):

    Phase 1: Familiarisation with the data
    Phase 2: Generating initial codes
    Phase 3: Searching for themes
    Phase 4: Reviewing themes
    Phase 5: Defining and naming themes
    Phase 6: Producing the report (visualisations + summary)

Coding decisions are declared explicitly in the CODES, CODED, and THEMES
dictionaries below. Each contribution is hand-mapped to one or more codes
with a verbatim evidence quote for audit-trail purposes — this matches
how the three semi-structured interviews were analysed earlier in the
project, so the methodological treatment is consistent across the
qualitative chapter.
"""

import os
import re
import textwrap
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# PATHS — output goes into a sibling 'focus_group_outputs/' folder so it
# doesn't clutter the source folder. Works in VS Code, terminals, or
# notebooks.
# ---------------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUT_DIR = os.path.join(BASE_DIR, "focus_group_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# PHASE 1 — FAMILIARISATION
# Transcript encoded inline. Each contribution stored as
#   (topic_id, topic_label, participant, verbatim_text)
# ---------------------------------------------------------------------------

TOPICS = {
    "M1": "General feelings on AI for stock prediction",
    "M2": "What current tools are missing",
    "M3": "Importance of news / sentiment",
    "M4": "Value of an integrated system",
    "M5": "Automation vs. user control",
    "M6": "Biggest issue with current tools",
}

PARTICIPANTS = ["Student", "Professional", "Expert"]

TRANSCRIPT = [
    # --- M1: General feelings ---
    ("M1", "Student",
     "I mean… I've seen it but I don't really trust it that much. Like "
     "it just shows numbers and I don't know how it's working."),
    ("M1", "Professional",
     "Yeah I kind of agree… I use some tools but I don't rely fully on "
     "them. It's more like an extra input, not something I'd base "
     "decisions on completely."),
    ("M1", "Expert",
     "That's fair. Even for me, I don't rely on AI tools directly. The "
     "issue is not just accuracy, it's that they don't explain the "
     "reasoning properly."),
    # --- M2: What's missing ---
    ("M2", "Student",
     "For me it's mostly understanding… like I don't know why something "
     "is going up or down."),
    ("M2", "Professional",
     "Yeah and also everything is separate. You check charts in one "
     "place, news somewhere else, and then you have to connect it "
     "yourself."),
    ("M2", "Expert",
     "Exactly. There's no proper integration. News, sentiment, market "
     "data… all of it is relevant, but you have to process it manually. "
     "That's the biggest gap."),
    # --- M3: News / sentiment ---
    ("M3", "Student",
     "I think it matters… like if something big happens it affects the "
     "price, right? But I don't know how to use that information."),
    ("M3", "Professional",
     "It definitely matters, especially for short-term moves. But tools "
     "don't really connect it clearly with the data."),
    ("M3", "Expert",
     "It's very important, but the problem is consistency. News can be "
     "noisy. You need a structured way to extract signal from it, "
     "otherwise it's just information overload."),
    # --- M4: Integrated system ---
    ("M4", "Student",
     "Yeah that would help a lot… if it explains things simply."),
    ("M4", "Professional",
     "Definitely. It would save time. Right now I spend more time trying "
     "to understand what's happening than actually making decisions."),
    ("M4", "Expert",
     "It would be valuable, but only if it's done properly. It needs to "
     "combine data, not just show it. And it has to explain why it's "
     "giving a certain output."),
    # --- M5: Automation vs. control ---
    ("M5", "Student",
     "I'd still want to understand it… not just follow blindly."),
    ("M5", "Professional",
     "Same… some level of control is important, but automation would "
     "help reduce effort."),
    ("M5", "Expert",
     "Full automation without transparency won't work. The system should "
     "assist decision-making, not replace it completely."),
    # --- M6: Biggest issue ---
    ("M6", "Student",
     "For me it's just confusing… I don't understand it."),
    ("M6", "Professional",
     "For me it's time and effort… too many sources to check."),
    ("M6", "Expert",
     "For me it's lack of integration and explainability. The tools are "
     "either too basic or too opaque. There's no system that brings "
     "everything together in a meaningful way."),
]

print("PHASE 1 — Familiarisation")
print("-" * 60)
print(f"  {len(TRANSCRIPT)} contributions across {len(TOPICS)} topics, "
      f"{len(PARTICIPANTS)} participants.")
total_words = sum(len(t.split()) for _, _, t in TRANSCRIPT)
print(f"  Corpus size: {total_words} words.")

# ---------------------------------------------------------------------------
# PHASE 2 — INITIAL CODES
# Each code captures one analytically meaningful idea grounded in the
# transcript. Codes are short, noun-phrased, and definitions kept tight.
# ---------------------------------------------------------------------------

CODES = {
    "C1":  ("Distrust of AI predictions",
            "Reluctance to fully trust AI outputs as a basis for decisions."),
    "C2":  ("Black-box / opacity",
            "Tool's reasoning is invisible; outputs without visible logic."),
    "C3":  ("Need for explainability",
            "Wish for the system to make its reasoning visible."),
    "C4":  ("Accuracy concern",
            "Predictions may not be accurate enough to act on."),
    "C5":  ("Fragmented tooling",
            "Information scattered across multiple separate platforms."),
    "C6":  ("Manual cross-source integration",
            "User has to combine data sources by hand."),
    "C7":  ("Lack of integration in current tools",
            "Tools fail to bring data, news and sentiment together."),
    "C8":  ("News / sentiment importance",
            "News and sentiment are seen as material to price moves."),
    "C9":  ("Sentiment-extraction difficulty",
            "Hard to translate noisy news/sentiment into a usable signal."),
    "C10": ("Openness to integrated AI system",
            "Would adopt a system that combines sources, if done well."),
    "C11": ("Time / effort burden",
            "Current process consumes too much of the user's time."),
    "C12": ("Conditional acceptance — explainability/integration",
            "Adoption is conditional on the system explaining itself "
            "and integrating sources properly."),
    "C13": ("Human-in-the-loop preference",
            "System should assist, not replace, the user."),
    "C14": ("AI as supplementary input only",
            "AI treated as one input among many, not the basis for "
            "decisions."),
    "C15": ("Knowledge / accessibility barrier",
            "User lacks the financial knowledge to interpret outputs."),
}

# Researcher-coded mapping. Each entry: (topic_id, participant) -> codes,
# with a verbatim evidence quote to support each coding decision.
CODED = {
    # ---- M1 ----
    ("M1", "Student"):      {"codes": ["C1", "C2", "C15"],
                             "evidence": "I don't really trust it that much "
                                         "... it just shows numbers and I "
                                         "don't know how it's working"},
    ("M1", "Professional"): {"codes": ["C14", "C1"],
                             "evidence": "I don't rely fully on them. It's "
                                         "more like an extra input"},
    ("M1", "Expert"):       {"codes": ["C2", "C3", "C4"],
                             "evidence": "the issue is not just accuracy, "
                                         "it's that they don't explain the "
                                         "reasoning properly"},
    # ---- M2 ----
    ("M2", "Student"):      {"codes": ["C3", "C15"],
                             "evidence": "I don't know why something is "
                                         "going up or down"},
    ("M2", "Professional"): {"codes": ["C5", "C6"],
                             "evidence": "everything is separate. You check "
                                         "charts in one place, news "
                                         "somewhere else, and then you have "
                                         "to connect it yourself"},
    ("M2", "Expert"):       {"codes": ["C7", "C6"],
                             "evidence": "no proper integration. News, "
                                         "sentiment, market data... all of "
                                         "it is relevant, but you have to "
                                         "process it manually"},
    # ---- M3 ----
    ("M3", "Student"):      {"codes": ["C8", "C15"],
                             "evidence": "if something big happens it "
                                         "affects the price ... but I don't "
                                         "know how to use that information"},
    ("M3", "Professional"): {"codes": ["C8", "C7"],
                             "evidence": "tools don't really connect it "
                                         "clearly with the data"},
    ("M3", "Expert"):       {"codes": ["C8", "C9"],
                             "evidence": "the problem is consistency. News "
                                         "can be noisy. You need a "
                                         "structured way to extract signal"},
    # ---- M4 ----
    ("M4", "Student"):      {"codes": ["C10", "C3"],
                             "evidence": "that would help a lot... if it "
                                         "explains things simply"},
    ("M4", "Professional"): {"codes": ["C10", "C11"],
                             "evidence": "It would save time. Right now I "
                                         "spend more time trying to "
                                         "understand what's happening"},
    ("M4", "Expert"):       {"codes": ["C12", "C3", "C7"],
                             "evidence": "valuable, but only if it's done "
                                         "properly... it has to explain "
                                         "why it's giving a certain output"},
    # ---- M5 ----
    ("M5", "Student"):      {"codes": ["C13", "C3"],
                             "evidence": "I'd still want to understand "
                                         "it... not just follow blindly"},
    ("M5", "Professional"): {"codes": ["C13", "C11"],
                             "evidence": "some level of control is "
                                         "important, but automation would "
                                         "help reduce effort"},
    ("M5", "Expert"):       {"codes": ["C13", "C12"],
                             "evidence": "Full automation without "
                                         "transparency won't work. The "
                                         "system should assist "
                                         "decision-making, not replace it"},
    # ---- M6 ----
    ("M6", "Student"):      {"codes": ["C2", "C15"],
                             "evidence": "it's just confusing... I don't "
                                         "understand it"},
    ("M6", "Professional"): {"codes": ["C5", "C11"],
                             "evidence": "time and effort... too many "
                                         "sources to check"},
    ("M6", "Expert"):       {"codes": ["C7", "C3", "C2"],
                             "evidence": "lack of integration and "
                                         "explainability. The tools are "
                                         "either too basic or too opaque"},
}

print(f"\nPHASE 2 — Initial codes")
print("-" * 60)
print(f"  {len(CODES)} codes generated, "
      f"{sum(len(v['codes']) for v in CODED.values())} code-applications "
      f"across {len(CODED)} contributions.")

# ---------------------------------------------------------------------------
# PHASES 3 & 4 — SEARCHING FOR AND REVIEWING THEMES
# Codes clustered into themes. Each code belongs to exactly one theme so
# the structure is auditable.
# ---------------------------------------------------------------------------

THEMES = {
    "T1 — Opacity & demand for explainability": {
        "codes":  ["C2", "C3"],
        "summary": "Across all three participants, current tools are "
                   "described as black boxes; explainability is the "
                   "single most repeated demand for any future system.",
    },
    "T2 — Fragmented tooling & manual integration": {
        "codes":  ["C5", "C6", "C7"],
        "summary": "Information lives in too many places. The Professional "
                   "and Expert in particular describe the burden of "
                   "manually stitching data, news and sentiment together.",
    },
    "T3 — News & sentiment as relevant but hard-to-use signal": {
        "codes":  ["C8", "C9"],
        "summary": "All three agree news and sentiment matter to prices, "
                   "but the Expert frames the problem as one of signal "
                   "extraction — news is noisy and needs a structured "
                   "treatment to be useful.",
    },
    "T4 — Trust deficit & cautious adoption": {
        "codes":  ["C1", "C4", "C14"],
        "summary": "AI is treated as a supplementary input rather than a "
                   "basis for decisions. Distrust is rooted partly in "
                   "accuracy concerns but more fundamentally in the "
                   "absence of visible reasoning.",
    },
    "T5 — Time / effort burden of current workflow": {
        "codes":  ["C11"],
        "summary": "The Professional in particular frames the current "
                   "workflow as time-expensive — research and "
                   "interpretation take more effort than the decision "
                   "itself.",
    },
    "T6 — Knowledge & accessibility barriers": {
        "codes":  ["C15"],
        "summary": "The Student repeatedly surfaces a knowledge barrier: "
                   "even when the tool produces an output, lack of "
                   "financial literacy makes it unusable.",
    },
    "T7 — Conditional appetite for an integrated, explainable, "
    "human-in-the-loop system": {
        "codes":  ["C10", "C12", "C13"],
        "summary": "All three participants would adopt a system that "
                   "combines sources, but adoption is explicitly "
                   "conditional on transparency, simplicity and the user "
                   "remaining in control of the final decision.",
    },
}

CODE_TO_THEME = {c: t for t, info in THEMES.items() for c in info["codes"]}
unassigned = [c for c in CODES if c not in CODE_TO_THEME]
assert not unassigned, f"Unassigned codes: {unassigned}"

print(f"\nPHASES 3 & 4 — Theme search and review")
print("-" * 60)
for tname, info in THEMES.items():
    print(f"  {tname}  ({len(info['codes'])} codes)")

# ---------------------------------------------------------------------------
# PHASE 6 — AGGREGATE, VISUALISE, REPORT
# ---------------------------------------------------------------------------

# Counts per (participant, code)
code_counts = {p: Counter() for p in PARTICIPANTS}
for (m, p), entry in CODED.items():
    for c in entry["codes"]:
        code_counts[p][c] += 1

# Counts per (participant, theme)
theme_counts = {p: Counter() for p in PARTICIPANTS}
for p in PARTICIPANTS:
    for c, n in code_counts[p].items():
        theme_counts[p][CODE_TO_THEME[c]] += n

theme_names = list(THEMES.keys())
theme_freq = np.array(
    [[theme_counts[p][t] for p in PARTICIPANTS] for t in theme_names]
)

# Counts per (topic, theme) — useful for showing how the discussion moved
topic_theme = {m: Counter() for m in TOPICS}
for (m, p), entry in CODED.items():
    for c in entry["codes"]:
        topic_theme[m][CODE_TO_THEME[c]] += 1

# Code totals across the whole focus group (used for dot sizing)
code_total = Counter()
for p in PARTICIPANTS:
    for c, n in code_counts[p].items():
        code_total[c] += n

# ---------- Plot styling (matches the interview & survey figures) ----------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "semibold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#444",
    "axes.labelcolor":   "#222",
    "xtick.color":       "#444",
    "ytick.color":       "#444",
    "axes.grid":         False,
})

PARTICIPANT_COLORS = {
    "Student":      "#5B8FB9",   # muted blue
    "Professional": "#B97A56",   # warm clay
    "Expert":       "#5C8C6B",   # sage green
}
THEME_CMAP = LinearSegmentedColormap.from_list(
    "warm_mono", ["#FBF7F2", "#D4A373", "#7B4A2E"]
)

# =========================================================================
# FIGURE 1 — Theme prevalence by participant (grouped bar chart)
# =========================================================================
fig, ax = plt.subplots(figsize=(11.5, 5.8))
x = np.arange(len(theme_names))
bar_w = 0.26
for i, p in enumerate(PARTICIPANTS):
    vals = [theme_counts[p][t] for t in theme_names]
    ax.bar(x + (i - 1) * bar_w, vals, bar_w,
           label=p, color=PARTICIPANT_COLORS[p],
           edgecolor="white", linewidth=0.8)

# Wrap long theme labels onto two lines
wrapped_labels = []
for t in theme_names:
    name = t.split(" — ", 1)[1] if " — " in t else t
    wrapped_labels.append(textwrap.fill(name, width=22))
ax.set_xticks(x)
ax.set_xticklabels(wrapped_labels, fontsize=8.8)
ax.set_ylabel("Code applications (count)")
ax.set_title("Theme prevalence by participant", loc="left",
             pad=14, weight="semibold")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0, max(theme_freq.max(), 4) + 1)
ax.grid(axis="y", linestyle=":", color="#bbb", alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "figure1_theme_prevalence.png")
plt.savefig(fig1_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()

# =========================================================================
# FIGURE 2 — Theme × participant heatmap
# =========================================================================
fig, ax = plt.subplots(figsize=(9.5, 6.0))
im = ax.imshow(theme_freq, cmap=THEME_CMAP, aspect="auto")
ax.set_xticks(range(len(PARTICIPANTS)))
ax.set_xticklabels(PARTICIPANTS, fontsize=10)
ax.set_yticks(range(len(theme_names)))
ax.set_yticklabels(theme_names, fontsize=10)
ax.tick_params(axis="x", pad=8, length=0)
ax.tick_params(axis="y", length=0)

vmax = theme_freq.max() if theme_freq.max() else 1
for i in range(theme_freq.shape[0]):
    for j in range(theme_freq.shape[1]):
        v = theme_freq[i, j]
        ax.text(j, i, str(v),
                ha="center", va="center",
                color="#222" if v <= vmax * 0.55 else "white",
                fontsize=11, weight="semibold")
ax.set_title("Theme × Participant — code-application frequency",
             loc="left", pad=14, weight="semibold")
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("Count", color="#444")
cbar.ax.tick_params(colors="#444")
plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "figure2_theme_heatmap.png")
plt.savefig(fig2_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()

# =========================================================================
# FIGURE 3 — Codebook structure (codes → themes, with per-participant counts)
# =========================================================================
ordered_codes = []
theme_palette = ["#7B4A2E", "#9C6644", "#B08968",
                 "#5C8C6B", "#5B8FB9", "#7E6B8F", "#A0522D"]
theme_color = {t: theme_palette[i % len(theme_palette)]
               for i, t in enumerate(theme_names)}
theme_row_span = {}

for tname in theme_names:
    codes_in_theme = THEMES[tname]["codes"]
    first = len(ordered_codes)
    ordered_codes.extend(codes_in_theme)
    last  = len(ordered_codes) - 1
    theme_row_span[tname] = (first, last)

n_rows = len(ordered_codes)
fig, ax = plt.subplots(figsize=(13, 0.55 * n_rows + 1.4))
ax.set_xlim(0, 14)
ax.set_ylim(-0.5, n_rows - 0.5)
ax.invert_yaxis()
ax.axis("off")

band_left, band_right = 0.2, 4.4
for tname, (a, b) in theme_row_span.items():
    color = theme_color[tname]
    height = (b - a) + 0.85
    y_top  = a - 0.42
    rect = mpatches.FancyBboxPatch(
        (band_left, y_top), band_right - band_left, height,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=0, facecolor=color, alpha=0.92)
    ax.add_patch(rect)
    parts = tname.split(" — ", 1)
    label_top = parts[0]
    label_bot = parts[1] if len(parts) > 1 else ""
    cy = y_top + height / 2
    ax.text(band_left + 0.18, cy - 0.18, label_top,
            color="white", fontsize=11, weight="bold", va="center")
    ax.text(band_left + 0.18, cy + 0.20,
            textwrap.fill(label_bot, width=28),
            color="white", fontsize=9.0, va="center")

dot_x  = 5.8
text_x = 6.2
for row, c in enumerate(ordered_codes):
    tname = CODE_TO_THEME[c]
    color = theme_color[tname]
    ax.plot([band_right, dot_x], [row, row],
            color=color, linewidth=1.3, alpha=0.65, solid_capstyle="round")
    size = 130 + code_total[c] * 60
    ax.scatter(dot_x, row, s=size, color=color, alpha=0.9,
               edgecolor="white", linewidth=1.5, zorder=3)
    label = f"{c} — {CODES[c][0]}"
    ax.text(text_x, row, label, fontsize=10, va="center", color="#222")
    pcounts = [code_counts[p][c] for p in PARTICIPANTS]
    badge = (f"n={code_total[c]}   "
             f"(S:{pcounts[0]}, P:{pcounts[1]}, E:{pcounts[2]})")
    ax.text(13.85, row, badge, fontsize=8.5, color="#666",
            va="center", ha="right", style="italic")

ax.text(band_left, -1.2, "Codebook structure: codes → themes",
        fontsize=14, weight="semibold", color="#222")
ax.text(band_left, -0.78,
        "Dot size reflects total code applications across the focus "
        "group.  Per-participant counts shown right "
        "(S = Student, P = Professional, E = Expert).",
        fontsize=8.8, color="#666", style="italic")

plt.subplots_adjust(top=0.93, bottom=0.02)
fig3_path = os.path.join(OUT_DIR, "figure3_codebook_structure.png")
plt.savefig(fig3_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()

# =========================================================================
# FIGURE 4 — Topic × theme heatmap (focus-group-specific)
# Shows how the discussion's emphasis shifted as the moderator moved
# through the six topics. This is the analytical view that interview
# data didn't permit, since interviews don't have a comparable shared
# topical structure across participants.
# =========================================================================
topic_ids = list(TOPICS.keys())
topic_matrix = np.array(
    [[topic_theme[m][t] for m in topic_ids] for t in theme_names]
)

fig, ax = plt.subplots(figsize=(11.5, 6.6))
im = ax.imshow(topic_matrix, cmap=THEME_CMAP, aspect="auto")
ax.set_xticks(range(len(topic_ids)))
# Two-line label: topic ID on top, tightly wrapped description below.
xtick_labels = [f"{m}\n{textwrap.fill(TOPICS[m], width=12)}"
                for m in topic_ids]
ax.set_xticklabels(xtick_labels, fontsize=8.0, linespacing=1.25)
ax.set_yticks(range(len(theme_names)))
ax.set_yticklabels(theme_names, fontsize=10)
ax.tick_params(axis="x", pad=6, length=0)
ax.tick_params(axis="y", length=0)

vmax = topic_matrix.max() if topic_matrix.max() else 1
for i in range(topic_matrix.shape[0]):
    for j in range(topic_matrix.shape[1]):
        v = topic_matrix[i, j]
        ax.text(j, i, str(v) if v else "",
                ha="center", va="center",
                color="#222" if v <= vmax * 0.55 else "white",
                fontsize=10.5, weight="semibold")
ax.set_title("Theme × Discussion topic — how the conversation moved",
             loc="left", pad=14, weight="semibold")
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("Code applications", color="#444")
cbar.ax.tick_params(colors="#444")
plt.tight_layout()
fig4_path = os.path.join(OUT_DIR, "figure4_topic_theme.png")
plt.savefig(fig4_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Text report (codebook, theme summaries, cross-case pattern, full audit)
# ---------------------------------------------------------------------------
report = []
def w(s=""): report.append(s)

w("THEMATIC ANALYSIS — FOCUS GROUP DISCUSSION")
w("=" * 72)
w("Method:        Reflexive thematic analysis (Braun & Clarke, 2006).")
w(f"Participants:  {', '.join(PARTICIPANTS)}.")
w(f"Format:        {len(TOPICS)} moderator-led topics, "
  f"{len(TRANSCRIPT)} contributions, {total_words} words.")
w(f"Codebook:      {len(CODES)} codes -> {len(THEMES)} themes.")
w("")

w("1. DISCUSSION TOPICS")
w("-" * 72)
for mid, label in TOPICS.items():
    w(f"  {mid}: {label}")
w("")

w("2. CODES")
w("-" * 72)
for cid, (label, definition) in CODES.items():
    w(f"  {cid:>4}  {label}")
    w(f"        Definition: {definition}")
    w(f"        Total applications: {code_total[cid]}")
w("")

w("3. THEMES")
w("-" * 72)
for tname, info in THEMES.items():
    total = sum(theme_counts[p][tname] for p in PARTICIPANTS)
    present = sum(1 for p in PARTICIPANTS if theme_counts[p][tname] > 0)
    w(f"  {tname}")
    w(f"      Codes:    {', '.join(info['codes'])}")
    w(f"      Coverage: present in {present}/{len(PARTICIPANTS)} participants; "
      f"{total} total code applications")
    w(f"      Summary:  {info['summary']}")
    w("")

w("4. CROSS-CASE PATTERN")
w("-" * 72)
universal = [t for t in theme_names
             if all(theme_counts[p][t] > 0 for p in PARTICIPANTS)]
divergent = [t for t in theme_names
             if 0 < sum(1 for p in PARTICIPANTS if theme_counts[p][t] > 0)
                      < len(PARTICIPANTS)]
w(f"  Themes present in ALL three participants ({len(universal)}):")
for t in universal: w(f"      • {t}")
w("")
w(f"  Themes present in some but not all ({len(divergent)}):")
for t in divergent:
    who = [p for p in PARTICIPANTS if theme_counts[p][t] > 0]
    w(f"      • {t}  —  appears in: {', '.join(who)}")
w("")

w("5. ILLUSTRATIVE EXTRACTS (audit trail)")
w("-" * 72)
for (m, p), entry in CODED.items():
    code_labels = ", ".join(entry["codes"])
    w(f"  [{m} | {p} | {code_labels}]")
    w(f"      \"{entry['evidence']}\"")
    w("")

report_path = os.path.join(OUT_DIR, "thematic_analysis_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\nPHASE 6 — Outputs written")
print("-" * 60)
print(f"  Figure 1 (theme prevalence):     {fig1_path}")
print(f"  Figure 2 (theme × participant):  {fig2_path}")
print(f"  Figure 3 (codebook structure):   {fig3_path}")
print(f"  Figure 4 (topic × theme):        {fig4_path}")
print(f"  Text report (with audit trail):  {report_path}")

print("\n--- At-a-glance theme coverage ---")
for tname in theme_names:
    counts = [f"{p[0]}:{theme_counts[p][tname]}" for p in PARTICIPANTS]
    short = tname.split(" — ", 1)[1] if " — " in tname else tname
    print(f"  {short:<58}  " + "  ".join(counts))

print("\nDone.")