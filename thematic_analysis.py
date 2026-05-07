"""
Thematic Analysis of Three Stakeholder Interviews
==================================================
Stakeholders: Student, Working Professional, Self-Employed (experienced investor)
Topic: Perceptions of AI-based stock prediction tools

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

The codebook (codes -> themes) is declared explicitly below so the
analytical decisions are transparent and reproducible. Each interview
extract is mapped by the researcher to one or more codes; codes are
then aggregated into themes for cross-case comparison.
"""

import os
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---------------------------------------------------------------------------
# PHASE 1 -- FAMILIARISATION
# Read each transcript and split into Q/A pairs. We retain participant
# identifiers, question numbers, and verbatim text for later coding.
# ---------------------------------------------------------------------------

# Resolve paths relative to this script's folder so it works wherever
# the project is opened (VS Code, terminal, another machine, etc.).
# Falls back to the current working directory if __file__ isn't defined
# (e.g. when pasted into a Jupyter cell).
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

INPUT_FILES = {
    "Student":       os.path.join(BASE_DIR, "Student Interview.txt"),
    "Working Prof.": os.path.join(BASE_DIR, "Work_Prof Interview.txt"),
    "Self-Employed": os.path.join(BASE_DIR, "Self_Interview.txt"),
}

# Helpful error if a file isn't found, instead of a cryptic traceback.
for name, path in INPUT_FILES.items():
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find '{os.path.basename(path)}' for participant "
            f"'{name}'.\n"
            f"Looked in: {BASE_DIR}\n"
            f"Place the three interview .txt files in the same folder as "
            f"this script, or edit INPUT_FILES to point at their location."
        )

def parse_interview(path):
    """Split an interview transcript into a list of (q_num, q_text, a_text)."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # Split on the '---' separator between Q/A blocks
    blocks = [b.strip() for b in raw.split("---") if b.strip()]
    parsed = []
    for b in blocks:
        m = re.match(r"Q(\d+)\.\s*(.+?)\n\n(.+)", b, flags=re.DOTALL)
        if m:
            qnum  = int(m.group(1))
            qtext = m.group(2).strip()
            atext = m.group(3).strip()
            parsed.append((qnum, qtext, atext))
    return parsed

interviews = {p: parse_interview(path) for p, path in INPUT_FILES.items()}

print("PHASE 1 — Familiarisation")
print("-" * 60)
for participant, qa in interviews.items():
    print(f"  {participant}: {len(qa)} responses, "
          f"{sum(len(a.split()) for _,_,a in qa)} words total")

# ---------------------------------------------------------------------------
# PHASE 2 -- INITIAL CODES (researcher-derived)
# Each code captures one analytically meaningful idea. Codes are short,
# noun-phrased, and grounded in the interview text. The 'evidence' string
# stored for each (participant, q_num, code) is the verbatim quote fragment
# that justified the code -- this is essential for audit-trail / viva.
# ---------------------------------------------------------------------------

# Codebook: code_id -> (short label, definition)
CODES = {
    "C1":  ("Black-box / opacity",
            "Tool's reasoning is invisible to the user; outputs without "
            "explanation."),
    "C2":  ("Need for explainability",
            "Desire for transparent reasoning behind a recommendation."),
    "C3":  ("Distrust of AI predictions",
            "Reluctance to fully trust AI-generated predictions."),
    "C4":  ("Accuracy uncertainty",
            "Unsure whether predictions are actually accurate."),
    "C5":  ("Fragmented tooling",
            "Information scattered across multiple separate tools/sources."),
    "C6":  ("Manual cross-source integration",
            "User has to combine data sources by hand."),
    "C7":  ("News integration gap",
            "Tools fail to incorporate news adequately."),
    "C8":  ("Sentiment integration gap",
            "Tools fail to incorporate market or social sentiment."),
    "C9":  ("Static / lack of real-time adaptability",
            "Tool does not respond to changing market conditions in real time."),
    "C10": ("Complexity / accessibility barrier",
            "Tool is too technical for non-experts."),
    "C11": ("Multi-signal decision approach",
            "Participant uses several signals/sources to form a view."),
    "C12": ("Openness to integrated AI system",
            "Would use a system that combines data + news + insight, "
            "if explained."),
    "C13": ("Desire for actionable insight",
            "Wants a recommendation that supports a decision, not raw data."),
    "C14": ("Time-saving expectation",
            "Sees automation as a way to save analytical effort."),
    "C15": ("Human-in-the-loop preference",
            "Wants the system to assist, not replace, the user."),
}

# Researcher-coded mapping. Each entry: (participant, q_num) -> [code_ids],
# with a verbatim evidence quote (used for audit trail and on-chart tooltips).
CODED = {
    # ---- Student ----------------------------------------------------------
    ("Student", 1): {
        "codes": ["C10"],
        "evidence": "they show graphs and numbers ... I don't really know "
                    "how to read them properly ... it felt like I was just "
                    "guessing",
    },
    ("Student", 2): {
        "codes": ["C3", "C2"],
        "evidence": "I don't know how it's making the prediction, so it "
                    "feels risky ... I'd want to know the reason behind it",
    },
    ("Student", 3): {
        "codes": ["C2", "C10"],
        "evidence": "needs to be explained in a much simpler way ... too "
                    "technical, like it's made for people who already "
                    "understand finance",
    },
    ("Student", 4): {
        "codes": ["C1", "C2"],
        "evidence": "it definitely feels like a black box ... I don't know "
                    "what data it's using or how it's coming to that "
                    "conclusion",
    },
    ("Student", 5): {
        "codes": ["C12", "C2", "C13"],
        "evidence": "I would use something like that ... if it tells me "
                    "what's happening in the market based on news ... but "
                    "still I would want some explanation",
    },

    # ---- Working Professional --------------------------------------------
    ("Working Prof.", 1): {
        "codes": ["C11"],
        "evidence": "I check some apps and also news regularly ... basic "
                    "charts and whatever I read online",
    },
    ("Working Prof.", 2): {
        "codes": ["C4", "C3", "C11"],
        "evidence": "I don't really know how accurate those AI predictions "
                    "are, so I treat them more like guidance rather than "
                    "something final",
    },
    ("Working Prof.", 3): {
        "codes": ["C7", "C5", "C6"],
        "evidence": "tools focus on charts and numbers but they don't "
                    "always include what's happening in the news properly "
                    "... I still have to check multiple sources",
    },
    ("Working Prof.", 4): {
        "codes": ["C12", "C13", "C14"],
        "evidence": "if something could combine that and give a clear "
                    "insight, it would save time and probably make "
                    "decisions easier",
    },
    ("Working Prof.", 5): {
        "codes": ["C15", "C2", "C12"],
        "evidence": "a balance would be better ... still show me enough "
                    "detail so I can understand what's happening ... not "
                    "fully automatic without any explanation",
    },

    # ---- Self-Employed (experienced investor) ----------------------------
    ("Self-Employed", 1): {
        "codes": ["C11"],
        "evidence": "a mix of things ... price action, broader market "
                    "trends, and also macro factors ... combining multiple "
                    "signals",
    },
    ("Self-Employed", 2): {
        "codes": ["C8", "C6", "C14"],
        "evidence": "Mostly manually ... no structured way I use sentiment "
                    "... it does take time and there's always some "
                    "subjectivity",
    },
    ("Self-Employed", 3): {
        "codes": ["C7", "C8", "C5"],
        "evidence": "most tools either focus on technical data or just "
                    "give some basic news feed, but they don't integrate "
                    "both properly",
    },
    ("Self-Employed", 4): {
        "codes": ["C2", "C1", "C9"],
        "evidence": "they don't explain their reasoning clearly ... markets "
                    "are dynamic, so a static model doesn't always capture "
                    "what's happening in real time",
    },
    ("Self-Employed", 5): {
        "codes": ["C12", "C13", "C2", "C15"],
        "evidence": "combine market data, news, and maybe even sentiment "
                    "in a structured way and give actionable insights with "
                    "explanation ... still allow the user to understand "
                    "the reasoning",
    },
}

print(f"\nPHASE 2 — Initial codes")
print("-" * 60)
print(f"  {len(CODES)} codes generated, "
      f"{sum(len(v['codes']) for v in CODED.values())} code-applications "
      f"across {len(CODED)} extracts.")

# ---------------------------------------------------------------------------
# PHASE 3 & 4 -- SEARCHING FOR AND REVIEWING THEMES
# Codes are clustered into broader themes. Each theme is a coherent pattern
# of meaning. The mapping is declared explicitly so it can be audited.
# ---------------------------------------------------------------------------

THEMES = {
    "T1 — Opacity & demand for explainability": {
        "codes":  ["C1", "C2"],
        "summary": "All three participants describe current tools as opaque "
                   "and explicitly call for the reasoning behind a prediction "
                   "to be made visible.",
    },
    "T2 — Fragmented tooling & manual integration": {
        "codes":  ["C5", "C6", "C11"],
        "summary": "Participants consult several disconnected tools and "
                   "perform the synthesis themselves, indicating a gap in "
                   "unified, integrated systems.",
    },
    "T3 — News & sentiment integration gap": {
        "codes":  ["C7", "C8", "C9"],
        "summary": "Existing tools are perceived as poorly equipped to "
                   "incorporate breaking news, public sentiment, and "
                   "real-time changes in market conditions.",
    },
    "T4 — Trust deficit & accuracy uncertainty": {
        "codes":  ["C3", "C4"],
        "summary": "Participants are unsure whether AI predictions are "
                   "accurate enough to act on, leading to cautious or "
                   "advisory use rather than decisive use.",
    },
    "T5 — Accessibility & complexity barriers": {
        "codes":  ["C10"],
        "summary": "Less experienced users are excluded by jargon-heavy "
                   "interfaces and assumed financial literacy.",
    },
    "T6 — Appetite for an integrated, actionable, explainable system": {
        "codes":  ["C12", "C13", "C14", "C15"],
        "summary": "Across all three profiles, participants would adopt a "
                   "system that combines sources, surfaces actionable "
                   "insights, saves time, and keeps the user informed and "
                   "in control.",
    },
}

# Reverse map: code -> theme (each code belongs to exactly one theme here)
CODE_TO_THEME = {c: t for t, info in THEMES.items() for c in info["codes"]}

# Sanity-check: every code is assigned to a theme
unassigned = [c for c in CODES if c not in CODE_TO_THEME]
assert not unassigned, f"Unassigned codes: {unassigned}"

print(f"\nPHASES 3 & 4 — Theme search and review")
print("-" * 60)
for tname, info in THEMES.items():
    print(f"  {tname}  ({len(info['codes'])} codes)")

# ---------------------------------------------------------------------------
# PHASE 5 -- DEFINING AND NAMING THEMES (already done above in THEMES dict)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PHASE 6 -- PRODUCING THE REPORT
# (a) Quantify code and theme frequency per participant
# (b) Build matrices for visualisation
# (c) Generate three output figures
# (d) Print a structured text report
# ---------------------------------------------------------------------------

participants = list(INPUT_FILES.keys())
code_ids     = list(CODES.keys())
theme_names  = list(THEMES.keys())

# Count per (participant, code)
code_counts = {p: Counter() for p in participants}
for (p, q), entry in CODED.items():
    for c in entry["codes"]:
        code_counts[p][c] += 1

# Count per (participant, theme)
theme_counts = {p: Counter() for p in participants}
for p in participants:
    for c, n in code_counts[p].items():
        theme_counts[p][CODE_TO_THEME[c]] += n

# Theme presence (binary): did this theme appear at least once for participant?
theme_present = np.array(
    [[1 if theme_counts[p][t] > 0 else 0 for p in participants]
     for t in theme_names]
)

# Theme frequency (count) matrix
theme_freq = np.array(
    [[theme_counts[p][t] for p in participants] for t in theme_names]
)

# ---------------- Output directory ----------------
# Write outputs to a sibling 'thematic_analysis_outputs' folder next to
# the script, so figures and the report don't clutter the source folder.
OUT_DIR = os.path.join(BASE_DIR, "thematic_analysis_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Aesthetic choices: warm muted academic palette, clean sans-serif
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#222",
    "xtick.color": "#444",
    "ytick.color": "#444",
    "axes.grid": False,
})

PARTICIPANT_COLORS = {
    "Student":        "#5B8FB9",   # muted blue
    "Working Prof.":  "#B97A56",   # warm clay
    "Self-Employed":  "#5C8C6B",   # sage green
}
THEME_CMAP = LinearSegmentedColormap.from_list(
    "warm_mono", ["#FBF7F2", "#D4A373", "#7B4A2E"]
)

# =========================================================================
# FIGURE 1 — Theme prevalence by participant (grouped bar chart)
# =========================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(theme_names))
bar_w = 0.26
for i, p in enumerate(participants):
    vals = [theme_counts[p][t] for t in theme_names]
    ax.bar(x + (i - 1) * bar_w, vals, bar_w,
           label=p, color=PARTICIPANT_COLORS[p],
           edgecolor="white", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([t.split(" — ")[1] for t in theme_names],
                   rotation=20, ha="right")
ax.set_ylabel("Code applications (count)")
ax.set_title("Theme prevalence by participant", loc="left",
             pad=14, weight="semibold")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0, max(theme_freq.max(), 4) + 1)
ax.grid(axis="y", linestyle=":", color="#bbb", alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
fig1_path = f"{OUT_DIR}/figure1_theme_prevalence.png"
plt.savefig(fig1_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()

# =========================================================================
# FIGURE 2 — Theme × participant heatmap (frequency)
# =========================================================================
fig, ax = plt.subplots(figsize=(9.5, 6.0))
im = ax.imshow(theme_freq, cmap=THEME_CMAP, aspect="auto")
ax.set_xticks(range(len(participants)))
ax.set_xticklabels(participants, fontsize=10)
ax.set_yticks(range(len(theme_names)))
ax.set_yticklabels(theme_names, fontsize=10)
# Move x-tick labels to the top for clarity, with breathing room
ax.xaxis.set_ticks_position("bottom")
ax.tick_params(axis="x", pad=8, length=0)
ax.tick_params(axis="y", length=0)
for i in range(theme_freq.shape[0]):
    for j in range(theme_freq.shape[1]):
        v = theme_freq[i, j]
        ax.text(j, i, str(v), ha="center", va="center",
                color="#222" if v <= 2 else "white", fontsize=11,
                weight="semibold")
ax.set_title("Theme × Participant — code-application frequency",
             loc="left", pad=14, weight="semibold")
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("Count", color="#444")
cbar.ax.tick_params(colors="#444")
plt.tight_layout()
fig2_path = f"{OUT_DIR}/figure2_theme_heatmap.png"
plt.savefig(fig2_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()

# =========================================================================
# FIGURE 3 — Code -> Theme hierarchy diagram
# Layout: one row per code (no overlaps possible), themes shown as colored
# grouping bands on the left that span the rows of their constituent codes.
# Dot size = total code-application count across the three interviews.
# =========================================================================

# Total counts per code (across participants)
code_total = Counter()
for p in participants:
    for c, n in code_counts[p].items():
        code_total[c] += n

# Build a per-code row order grouped by theme (preserves theme order)
ordered_codes = []           # list of code_ids, top to bottom
theme_palette = ["#7B4A2E", "#9C6644", "#B08968",
                 "#5C8C6B", "#5B8FB9", "#7E6B8F"]
theme_color = {t: theme_palette[i % len(theme_palette)]
               for i, t in enumerate(theme_names)}
theme_row_span = {}          # theme -> (first_row, last_row)

for tname in theme_names:
    codes_in_theme = THEMES[tname]["codes"]
    first = len(ordered_codes)
    ordered_codes.extend(codes_in_theme)
    last  = len(ordered_codes) - 1
    theme_row_span[tname] = (first, last)

n_rows = len(ordered_codes)

fig, ax = plt.subplots(figsize=(12, 0.55 * n_rows + 1.2))
ax.set_xlim(0, 14)
ax.set_ylim(-0.5, n_rows - 0.5)
ax.invert_yaxis()      # row 0 at top
ax.axis("off")

# ---- Left band: themes, spanning their code rows ----
band_left, band_right = 0.2, 4.2
for tname, (a, b) in theme_row_span.items():
    color = theme_color[tname]
    height = (b - a) + 0.85
    y_top  = a - 0.42
    rect = mpatches.FancyBboxPatch(
        (band_left, y_top), band_right - band_left, height,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=0, facecolor=color, alpha=0.92)
    ax.add_patch(rect)
    # Wrap theme name nicely (split at em-dash)
    parts = tname.split(" — ")
    label_top = parts[0]                       # "T1"
    label_bot = parts[1] if len(parts) > 1 else ""
    # Two-line label, centered vertically in the band
    cy = (y_top + height / 2)
    ax.text(band_left + 0.18, cy - 0.18, label_top,
            color="white", fontsize=11, weight="bold", va="center")
    # Word-wrap the theme description so it fits the band height
    import textwrap
    wrapped = textwrap.fill(label_bot, width=26)
    ax.text(band_left + 0.18, cy + 0.20, wrapped,
            color="white", fontsize=9.2, va="center")

# ---- Connectors + dots + code labels ----
dot_x  = 5.6
text_x = 6.0
for row, c in enumerate(ordered_codes):
    tname = CODE_TO_THEME[c]
    color = theme_color[tname]
    # Connector
    ax.plot([band_right, dot_x], [row, row],
            color=color, linewidth=1.3, alpha=0.65, solid_capstyle="round")
    # Dot, sized by total count
    size = 130 + code_total[c] * 80
    ax.scatter(dot_x, row, s=size, color=color, alpha=0.9,
               edgecolor="white", linewidth=1.5, zorder=3)
    # Code label
    label = f"{c} — {CODES[c][0]}"
    ax.text(text_x, row, label, fontsize=10, va="center", color="#222")
    # Per-participant counts on the right
    pcounts = [code_counts[p][c] for p in participants]
    badge_text = f"n={code_total[c]}   "  \
                 f"(S:{pcounts[0]}, W:{pcounts[1]}, SE:{pcounts[2]})"
    ax.text(13.85, row, badge_text, fontsize=8.5, color="#666",
            va="center", ha="right", style="italic")

# ---- Title block ----
ax.text(band_left, -1.1,
        "Codebook structure: codes → themes",
        fontsize=14, weight="semibold", color="#222")
ax.text(band_left, -0.7,
        "Dot size reflects total code applications across the three "
        "interviews. "
        "Per-participant counts shown right (S = Student, W = Working "
        "Prof., SE = Self-Employed).",
        fontsize=8.8, color="#666", style="italic")

plt.subplots_adjust(top=0.93, bottom=0.02)
fig3_path = f"{OUT_DIR}/figure3_codebook_structure.png"
plt.savefig(fig3_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Text report (saved alongside the figures)
# ---------------------------------------------------------------------------
report_lines = []
def w(s=""): report_lines.append(s)

w("THEMATIC ANALYSIS REPORT")
w("=" * 72)
w("Method: Reflexive thematic analysis (Braun & Clarke, 2006).")
w(f"Corpus: {len(participants)} semi-structured interviews "
  f"(Student, Working Professional, Self-Employed investor).")
w(f"Codebook: {len(CODES)} codes -> {len(THEMES)} themes.")
w("")

w("1. CODES")
w("-" * 72)
for cid, (label, definition) in CODES.items():
    w(f"  {cid:>4}  {label}")
    w(f"        Definition: {definition}")
    w(f"        Total applications: {code_total[cid]}")
w("")

w("2. THEMES")
w("-" * 72)
for tname, info in THEMES.items():
    total = sum(theme_counts[p][tname] for p in participants)
    present = sum(1 for p in participants if theme_counts[p][tname] > 0)
    w(f"  {tname}")
    w(f"      Codes:    {', '.join(info['codes'])}")
    w(f"      Coverage: present in {present}/{len(participants)} interviews; "
      f"{total} total code applications")
    w(f"      Summary:  {info['summary']}")
    w("")

w("3. CROSS-CASE PATTERN")
w("-" * 72)
ubiquitous = [t for t in theme_names
              if all(theme_counts[p][t] > 0 for p in participants)]
divergent  = [t for t in theme_names
              if 0 < sum(1 for p in participants if theme_counts[p][t] > 0)
                       < len(participants)]
w(f"  Themes present in ALL three interviews ({len(ubiquitous)}):")
for t in ubiquitous: w(f"      • {t}")
w("")
w(f"  Themes present in some but not all ({len(divergent)}):")
for t in divergent:
    who = [p for p in participants if theme_counts[p][t] > 0]
    w(f"      • {t}  —  appears in: {', '.join(who)}")
w("")

w("4. ILLUSTRATIVE EXTRACTS (audit trail)")
w("-" * 72)
for (p, q), entry in CODED.items():
    code_labels = ", ".join(f"{c}" for c in entry["codes"])
    w(f"  [{p} | Q{q} | {code_labels}]")
    w(f"      \"{entry['evidence']}\"")
    w("")

report_path = f"{OUT_DIR}/thematic_analysis_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nPHASE 6 — Outputs written")
print("-" * 60)
print(f"  Figure 1 (theme prevalence):  {fig1_path}")
print(f"  Figure 2 (theme heatmap):     {fig2_path}")
print(f"  Figure 3 (codebook structure):{fig3_path}")
print(f"  Text report:                  {report_path}")
print("\nDone.")