"""
Thematic Analysis of Open-Ended Survey Responses
=================================================
Three open-ended questions from the dissertation survey:
    Q13 — What do you feel is currently missing in AI-based stock prediction tools?
    Q14 — What would make AI-based stock predictions more useful or easier to rely on?
    Q15 — If you could design your ideal system for stock predictions,
          what would it look like?

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

Methodological note
-------------------
With 108 short (typically single-sentence) survey responses, hand-coding
every extract — as was appropriate for the three interviews — would be
impractical and inconsistent. Instead, the codebook below defines each
code by (i) a researcher-written definition and (ii) a set of keyword
patterns derived from a close reading of the corpus. Codes are then
applied programmatically via case-insensitive, word-boundary regex
matches. This keeps the *interpretive* step (deciding which patterns
of meaning matter) with the researcher, while making the *application*
step transparent, reproducible and auditable.

The full audit trail — every response, every code applied to it, every
matched keyword — is written to the report file.
"""

import os
import re
import csv
import textwrap
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# PATHS — resolved relative to the script so it works in VS Code, terminals,
# notebooks, or wherever the project is opened.
# ---------------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

INPUT_FILES = {
    "Q13 — Currently missing":  os.path.join(BASE_DIR, "survey", "Q13_currently_missing.csv"),
    "Q14 — More useful":        os.path.join(BASE_DIR, "survey", "Q14_more_useful.csv"),
    "Q15 — Ideal system":       os.path.join(BASE_DIR, "survey", "Q15_ideal_system.csv"),
}

for name, path in INPUT_FILES.items():
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find '{os.path.basename(path)}' for {name}.\n"
            f"Looked in: {BASE_DIR}\n"
            f"Place the three Q13/Q14/Q15 CSV files in the same folder as "
            f"this script, or edit INPUT_FILES to point at their location."
        )


# ---------------------------------------------------------------------------
# PHASE 1 -- FAMILIARISATION
# Read each CSV. Row 1 is the question text (header); subsequent rows are
# individual responses, one per row.
# ---------------------------------------------------------------------------

def load_responses(path):
    """Return (question_text, list_of_response_strings)."""
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    question = rows[0][0].strip()
    responses = [r[0].strip() for r in rows[1:] if r and r[0].strip()]
    return question, responses

questions = {}        # display_label -> full question text
responses = {}        # display_label -> list of response strings
for label, path in INPUT_FILES.items():
    q, rs = load_responses(path)
    questions[label] = q
    responses[label] = rs

print("PHASE 1 — Familiarisation")
print("-" * 60)
for label, rs in responses.items():
    n_words = sum(len(r.split()) for r in rs)
    print(f"  {label}: {len(rs)} responses, {n_words} words "
          f"(mean ~{n_words / max(len(rs),1):.1f} words/response)")


# ---------------------------------------------------------------------------
# PHASE 2 -- INITIAL CODES
# Each code: (label, definition, keyword pattern list).
# Patterns are matched case-insensitively with word boundaries; multi-word
# patterns are matched as phrases. Tweak the keywords to refine the
# codebook — the rest of the script picks up changes automatically.
# ---------------------------------------------------------------------------

CODES = {
    "C1": {
        "label": "Real-time / immediacy",
        "definition": "Calls for predictions or updates that occur as "
                      "events happen, with no perceptible delay.",
        "keywords": [
            r"real[\-\s]?time", r"instantly", r"instant",
            r"immediate(?:ly)?", r"as (?:they|new|it) (?:happen|happens)",
            r"as new", r"live events?", r"in real[\-\s]time",
            r"without delays?",
        ],
    },
    "C2": {
        "label": "Latency / lag criticism",
        "definition": "Critiques of existing tools as slow, late, or "
                      "behind real conditions.",
        "keywords": [
            r"\blag(?:s|ged|ging|ged behind|ging behind)?\b", r"\bdelayed?\b",
            r"\bbehind\b", r"\bslow(?:er|ly)?\b", r"\bcatch(?:es|ing)? up\b",
            r"\bafter (?:trends|the impact|the trend|hours)",
            r"\bafter trends form\b", r"\bhours later\b",
            r"\boutdated\b", r"\bmiss(?:es|ed|ing)? key (?:moments|triggers)\b",
        ],
    },
    "C3": {
        "label": "Adaptive / dynamic behaviour",
        "definition": "Wish for systems that learn, evolve, adapt or "
                      "update continuously rather than running on fixed rules.",
        "keywords": [
            r"\badapt(?:s|ive|ing|ed|ively)?\b", r"\bdynamic(?:ally)?\b",
            r"\bevolve(?:s|d)?\b", r"\bevolving\b",
            r"\blearn(?:s|ing|ed)?\b", r"\bcontinuous(?:ly)?\b",
            r"\bkeeps? (?:updating|learning|tracking)\b",
            r"\bupdates? continuously\b", r"\badjust(?:s|ed|ing|ments?)?\b",
        ],
    },
    "C4": {
        "label": "Static / rigid criticism",
        "definition": "Critiques of existing tools as static, rigid, "
                      "or built on fixed/historical patterns.",
        "keywords": [
            r"\bstatic\b", r"\brigid\b", r"\bfixed\b",
            r"\bfixed models?\b", r"\bfixed patterns?\b",
            r"\bfixed predictions?\b", r"\bsticking\b",
            r"\bpast trends?\b", r"\bhistorical patterns?\b",
            r"\brely(?:ing)? (?:only|heavily) on past\b",
        ],
    },
    "C5": {
        "label": "News & external event awareness",
        "definition": "Need for the system to incorporate news, "
                      "external events, developments or sudden shifts.",
        "keywords": [
            r"\bnews\b", r"\bevents?\b", r"\bdevelopments?\b",
            r"\breal[\-\s]world\b", r"\btriggers?\b",
            r"\bexternal influences?\b", r"\bsudden (?:shifts?|changes?)\b",
            r"\bunexpected events?\b", r"\bmarket reactions?\b",
            r"\bbreaking\b", r"\bongoing events?\b",
        ],
    },
    "C6": {
        "label": "Context-awareness",
        "definition": "Wish for the system to grasp what is currently "
                      "going on in the market or world around the data.",
        "keywords": [
            r"\bcontext(?:[\-\s]aware)?\b", r"\bcurrent situations?\b",
            r"\bcurrent developments?\b", r"\bcurrent (?:market )?conditions?\b",
            r"\bcurrent context\b", r"\bongoing developments?\b",
            r"\bawareness\b", r"\bcurrent\b",
        ],
    },
    "C7": {
        "label": "Multi-source integration",
        "definition": "Combining several inputs (data, news, sentiment, "
                      "external triggers) into one coherent system.",
        "keywords": [
            r"\bcombin(?:e|es|ed|ing)\b", r"\bintegrat(?:e|es|ed|ion|ing)\b",
            r"\bmultiple (?:inputs?|sources?|signals?)\b",
            r"\bdifferent signals?\b",
            r"\bdata and sentiment\b", r"\bnumbers and market\b",
            r"\bdata,? trends,? and external\b",
            r"\bboth (?:data|numbers|historical)\b",
        ],
    },
    "C8": {
        "label": "Explainability / reasoning",
        "definition": "Want the system to make its reasoning visible — "
                      "why it produced a given output or prediction.",
        "keywords": [
            r"\bexplain(?:s|ed|ing|ation)?\b", r"\breasoning\b",
            r"\bclarity\b", r"\binterpretation\b",
            r"\bshows? reasoning\b", r"\bexplains? why\b",
        ],
    },
    "C9": {
        "label": "Trust / reliability / confidence",
        "definition": "Outcomes the participant frames in terms of "
                      "trust, reliability or confidence in predictions.",
        "keywords": [
            r"\btrust(?:s|ed|worthy)?\b", r"\bconfidence\b",
            r"\bconfident\b", r"\breliab(?:le|ility|ly)\b",
        ],
    },
    "C10": {
        "label": "Background / hands-off automation",
        "definition": "System should run on its own — automatic updates, "
                      "background operation, minimal manual oversight.",
        "keywords": [
            r"\bautomatic(?:ally)?\b", r"\bwithout manual (?:intervention|checks?)\b",
            r"\bin the background\b", r"\bbackground\b",
            r"\bno (?:manual|human) (?:checks?|intervention)\b",
        ],
    },
    "C11": {
        "label": "Track record / historical validation",
        "definition": "Desire to see how a tool has performed historically, "
                      "especially through past downturns.",
        "keywords": [
            r"\btrack record\b", r"\bperformed in past\b",
            r"\bdownturns?\b", r"\bhistory of performance\b",
        ],
    },
    "C12": {
        "label": "Responsiveness to change",
        "definition": "System should respond to changes in inputs or "
                      "conditions, beyond just refreshing on a schedule.",
        "keywords": [
            r"\bresponsive(?:ness)?\b", r"\brespond(?:s|ed|ing)?\b",
            r"\breact(?:s|ed|ing|ion|ions)?\b",
        ],
    },
}

# Pre-compile patterns for performance and to surface any regex errors early.
COMPILED = {
    cid: [re.compile(p, flags=re.IGNORECASE) for p in info["keywords"]]
    for cid, info in CODES.items()
}

print(f"\nPHASE 2 — Initial codes")
print("-" * 60)
print(f"  {len(CODES)} codes defined, "
      f"{sum(len(c['keywords']) for c in CODES.values())} keyword patterns "
      f"in total.")


# ---------------------------------------------------------------------------
# PHASES 3 & 4 -- SEARCHING FOR AND REVIEWING THEMES
# Codes are clustered into themes that capture broader patterns of meaning.
# Each code belongs to exactly one theme.
# ---------------------------------------------------------------------------

THEMES = {
    "T1 — Real-time responsiveness & latency": {
        "codes": ["C1", "C2", "C12"],
        "summary": "Participants repeatedly frame the gap as a timing "
                   "problem: tools update too slowly, lag behind events, "
                   "and need to respond as conditions change.",
    },
    "T2 — Adaptive vs. static behaviour": {
        "codes": ["C3", "C4"],
        "summary": "Existing tools are described as rigid or stuck on "
                   "fixed patterns; participants want systems that adapt "
                   "and evolve continuously.",
    },
    "T3 — News & external event awareness": {
        "codes": ["C5"],
        "summary": "A specific class of inputs — news, real-world events, "
                   "sudden shifts — is repeatedly named as something "
                   "current tools fail to absorb.",
    },
    "T4 — Context & multi-source integration": {
        "codes": ["C6", "C7"],
        "summary": "Beyond raw data, participants ask for awareness of "
                   "current context and for multiple input streams "
                   "(data, sentiment, external triggers) to be combined.",
    },
    "T5 — Explainability & reasoning": {
        "codes": ["C8"],
        "summary": "An outcome-side concern: even if predictions are "
                   "right, participants want the reasoning behind them "
                   "to be visible.",
    },
    "T6 — Trust, reliability & track record": {
        "codes": ["C9", "C11"],
        "summary": "The vocabulary of trust, confidence and reliability "
                   "recurs as the framing for whether a tool is usable, "
                   "with one explicit call for historical track-record "
                   "validation.",
    },
    "T7 — Background, hands-off automation": {
        "codes": ["C10"],
        "summary": "A minority preference for systems that operate "
                   "automatically in the background, surfacing only the "
                   "final output for review.",
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
# PHASE 6 -- APPLY CODES TO RESPONSES, AGGREGATE, VISUALISE, REPORT
# ---------------------------------------------------------------------------

def apply_codes(response):
    """Return {code_id: [matched_keyword, ...]} for one response."""
    hits = {}
    for cid, patterns in COMPILED.items():
        matched = []
        for p in patterns:
            m = p.search(response)
            if m:
                matched.append(m.group(0))
        if matched:
            hits[cid] = matched
    return hits

# Apply codes to every response in every question
coded = defaultdict(list)   # question_label -> list of (response, {code: [kw,...]})
for label, rs in responses.items():
    for r in rs:
        coded[label].append((r, apply_codes(r)))

# Aggregations -------------------------------------------------------------
question_labels = list(INPUT_FILES.keys())
code_ids        = list(CODES.keys())
theme_names     = list(THEMES.keys())

# Code counts: how many responses in each question contain at least one
# match for each code? (Counting once per response avoids double-weighting
# responses that re-use the same keyword.)
code_counts = {q: Counter() for q in question_labels}
for q in question_labels:
    for _resp, hits in coded[q]:
        for cid in hits:
            code_counts[q][cid] += 1

# Theme counts (sum of constituent code counts per question)
theme_counts = {q: Counter() for q in question_labels}
for q in question_labels:
    for cid, n in code_counts[q].items():
        theme_counts[q][CODE_TO_THEME[cid]] += n

# Theme presence (how many responses touch the theme at all, deduped per response)
theme_response_share = {q: Counter() for q in question_labels}
for q in question_labels:
    for _resp, hits in coded[q]:
        themes_touched = {CODE_TO_THEME[c] for c in hits}
        for t in themes_touched:
            theme_response_share[q][t] += 1

# Frequency matrix used for the heatmap (responses touching theme)
theme_share_matrix = np.array(
    [[theme_response_share[q][t] for q in question_labels] for t in theme_names]
)

# Diagnostic: responses that received no codes
uncoded = {q: [r for r, h in coded[q] if not h] for q in question_labels}


# ---------------- Output directory ----------------
OUT_DIR = os.path.join(BASE_DIR, "survey_thematic_analysis_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- Plot styling (matches earlier dissertation figures) ------
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

QUESTION_COLORS = {
    "Q13 — Currently missing":  "#B97A56",   # warm clay
    "Q14 — More useful":        "#5B8FB9",   # muted blue
    "Q15 — Ideal system":       "#5C8C6B",   # sage green
}
THEME_CMAP = LinearSegmentedColormap.from_list(
    "warm_mono", ["#FBF7F2", "#D4A373", "#7B4A2E"]
)


# =========================================================================
# FIGURE 1 — Theme prevalence by question
# Y-axis: number of responses in that question that touch the theme.
# (Out of 36 per question.)
# =========================================================================
fig, ax = plt.subplots(figsize=(11.5, 5.8))
x = np.arange(len(theme_names))
bar_w = 0.26
for i, q in enumerate(question_labels):
    vals = [theme_response_share[q][t] for t in theme_names]
    ax.bar(x + (i - 1) * bar_w, vals, bar_w,
           label=q, color=QUESTION_COLORS[q],
           edgecolor="white", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([t.split(" — ")[1] for t in theme_names],
                   rotation=18, ha="right")
ax.set_ylabel("Responses touching the theme  (out of 36 per question)")
ax.set_title("Theme prevalence by survey question", loc="left",
             pad=14, weight="semibold")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0, max(theme_share_matrix.max(), 5) + 2)
ax.grid(axis="y", linestyle=":", color="#bbb", alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "figure1_theme_prevalence.png")
plt.savefig(fig1_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()


# =========================================================================
# FIGURE 2 — Theme × question heatmap (response-level frequency)
# =========================================================================
fig, ax = plt.subplots(figsize=(9.5, 6.0))
im = ax.imshow(theme_share_matrix, cmap=THEME_CMAP, aspect="auto")
ax.set_xticks(range(len(question_labels)))
ax.set_xticklabels(question_labels, fontsize=9.5)
ax.set_yticks(range(len(theme_names)))
ax.set_yticklabels(theme_names, fontsize=10)
ax.tick_params(axis="x", pad=8, length=0)
ax.tick_params(axis="y", length=0)

vmax = theme_share_matrix.max() if theme_share_matrix.max() else 1
for i in range(theme_share_matrix.shape[0]):
    for j in range(theme_share_matrix.shape[1]):
        v = theme_share_matrix[i, j]
        ax.text(j, i, str(v),
                ha="center", va="center",
                color="#222" if v <= vmax * 0.55 else "white",
                fontsize=11, weight="semibold")
ax.set_title("Theme × Question — number of responses touching each theme",
             loc="left", pad=14, weight="semibold")
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("Responses (of 36)", color="#444")
cbar.ax.tick_params(colors="#444")
plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "figure2_theme_heatmap.png")
plt.savefig(fig2_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()


# =========================================================================
# FIGURE 3 — Codebook structure (codes → themes)
# One row per code; theme-colored band on the left spans its codes.
# Dot size = total responses in which the code appears (across all 3 Qs).
# =========================================================================
code_total = Counter()
for q in question_labels:
    for cid, n in code_counts[q].items():
        code_total[cid] += n

ordered_codes  = []
theme_palette  = ["#7B4A2E", "#9C6644", "#B08968",
                  "#5C8C6B", "#5B8FB9", "#7E6B8F", "#A0522D"]
theme_color    = {t: theme_palette[i % len(theme_palette)]
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
    parts = tname.split(" — ")
    label_top = parts[0]
    label_bot = parts[1] if len(parts) > 1 else ""
    cy = (y_top + height / 2)
    ax.text(band_left + 0.18, cy - 0.18, label_top,
            color="white", fontsize=11, weight="bold", va="center")
    wrapped = textwrap.fill(label_bot, width=28)
    ax.text(band_left + 0.18, cy + 0.20, wrapped,
            color="white", fontsize=9.2, va="center")

dot_x  = 5.8
text_x = 6.2
for row, c in enumerate(ordered_codes):
    tname = CODE_TO_THEME[c]
    color = theme_color[tname]
    ax.plot([band_right, dot_x], [row, row],
            color=color, linewidth=1.3, alpha=0.65, solid_capstyle="round")
    size = 130 + code_total[c] * 10
    ax.scatter(dot_x, row, s=size, color=color, alpha=0.9,
               edgecolor="white", linewidth=1.5, zorder=3)
    label = f"{c} — {CODES[c]['label']}"
    ax.text(text_x, row, label, fontsize=10, va="center", color="#222")
    pcounts = [code_counts[q][c] for q in question_labels]
    badge = (f"n={code_total[c]}   "
             f"(Q13:{pcounts[0]}, Q14:{pcounts[1]}, Q15:{pcounts[2]})")
    ax.text(13.85, row, badge, fontsize=8.5, color="#666",
            va="center", ha="right", style="italic")

ax.text(band_left, -1.2, "Codebook structure: codes → themes",
        fontsize=14, weight="semibold", color="#222")
ax.text(band_left, -0.78,
        "Dot size reflects total response-level appearances across the "
        "three open-ended questions.  Per-question counts shown right.",
        fontsize=8.8, color="#666", style="italic")

plt.subplots_adjust(top=0.93, bottom=0.02)
fig3_path = os.path.join(OUT_DIR, "figure3_codebook_structure.png")
plt.savefig(fig3_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()


# ---------------------------------------------------------------------------
# TEXT REPORT (codebook, theme summaries, cross-question pattern, audit trail)
# ---------------------------------------------------------------------------
report = []
def w(s=""): report.append(s)

w("THEMATIC ANALYSIS — OPEN-ENDED SURVEY RESPONSES")
w("=" * 72)
w("Method: Reflexive thematic analysis (Braun & Clarke, 2006), with "
  "researcher-defined codes applied via keyword pattern matching for "
  "transparency at the application step.")
w(f"Corpus: {sum(len(rs) for rs in responses.values())} responses across "
  f"{len(responses)} open-ended questions.")
w(f"Codebook: {len(CODES)} codes -> {len(THEMES)} themes.")
w("")

w("1. QUESTIONS")
w("-" * 72)
for label, qtext in questions.items():
    w(f"  {label}")
    w(f"      \"{qtext}\"")
    w(f"      Responses: {len(responses[label])}")
w("")

w("2. CODES")
w("-" * 72)
for cid, info in CODES.items():
    w(f"  {cid:>4}  {info['label']}")
    w(f"        Definition: {info['definition']}")
    w(f"        Patterns:   {', '.join(info['keywords'])}")
    w(f"        Total response-level appearances: {code_total[cid]}")
w("")

w("3. THEMES")
w("-" * 72)
for tname, info in THEMES.items():
    total = sum(theme_response_share[q][tname] for q in question_labels)
    w(f"  {tname}")
    w(f"      Codes:        {', '.join(info['codes'])}")
    w(f"      Coverage:     {total} responses across the three questions "
      f"(out of {sum(len(rs) for rs in responses.values())})")
    for q in question_labels:
        share = theme_response_share[q][tname]
        n = len(responses[q])
        pct = share / n * 100 if n else 0
        w(f"        - {q}: {share}/{n}  ({pct:.0f}%)")
    w(f"      Summary:      {info['summary']}")
    w("")

w("4. CROSS-QUESTION PATTERN")
w("-" * 72)
def is_universal(t): return all(theme_response_share[q][t] >= 1 for q in question_labels)
def is_dominant(t):  return all(theme_response_share[q][t] / len(responses[q]) >= 0.30
                                for q in question_labels)

universal = [t for t in theme_names if is_universal(t)]
dominant  = [t for t in theme_names if is_dominant(t)]
w(f"  Themes appearing in all three questions ({len(universal)}):")
for t in universal: w(f"      • {t}")
w("")
w(f"  Themes touching ≥30% of responses in every question ({len(dominant)}):")
for t in dominant: w(f"      • {t}")
w("")

w("5. UNCODED RESPONSES (audit / coverage diagnostic)")
w("-" * 72)
for q in question_labels:
    w(f"  {q}: {len(uncoded[q])} responses received no code "
      f"(out of {len(responses[q])})")
    for r in uncoded[q]:
        w(f"      - \"{r}\"")
w("")

w("6. AUDIT TRAIL — every response with applied codes and matched keywords")
w("-" * 72)
for q in question_labels:
    w(f"\n  ── {q} ──")
    for i, (resp, hits) in enumerate(coded[q], 1):
        if not hits:
            w(f"  [{i:02d}] (no codes)   \"{resp}\"")
            continue
        # Compose a compact code+keyword summary
        bits = []
        for cid, kws in hits.items():
            kws_clean = ", ".join(sorted(set(kws)))
            bits.append(f"{cid}[{kws_clean}]")
        w(f"  [{i:02d}] {' '.join(bits)}")
        w(f"       \"{resp}\"")

report_path = os.path.join(OUT_DIR, "thematic_analysis_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"\nPHASE 6 — Outputs written")
print("-" * 60)
print(f"  Figure 1 (theme prevalence):    {fig1_path}")
print(f"  Figure 2 (theme heatmap):       {fig2_path}")
print(f"  Figure 3 (codebook structure):  {fig3_path}")
print(f"  Text report (with audit trail): {report_path}")

# Quick at-a-glance findings
print("\n--- At-a-glance theme coverage (responses touching theme) ---")
for tname in theme_names:
    parts = [f"{q.split(' — ')[0]} {theme_response_share[q][tname]:>2}/{len(responses[q])}"
             for q in question_labels]
    print(f"  {tname:<55}  " + "   ".join(parts))

# Diagnostic: how many responses got at least one code in each Q
print("\n--- Coverage diagnostic ---")
for q in question_labels:
    coded_n = sum(1 for _, h in coded[q] if h)
    n = len(responses[q])
    print(f"  {q}: {coded_n}/{n} responses received ≥1 code "
          f"({coded_n/n*100:.0f}%)")

print("\nDone.")