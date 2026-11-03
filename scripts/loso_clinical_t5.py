"""T5 — relate LOSO child-level errors to CARS and age (proposal2 §7.3), run
from train/ after loso_pooled_t5.py: `uv run python scripts/loso_clinical_t5.py`

Child score = mean P(ASD) over the child's held-out images
(artifacts/loso/loso_pooled_predictions.csv). CARS exists only for ASD
children, so severity is tested within ASD; age within each group. Four
exploratory tests, no multiplicity correction -- report as descriptive.

Output: artifacts/loso/loso_clinical.md
"""
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

META = "../../data/mahmoud-dataset/Metadata_Participants.csv"
PRED = "artifacts/loso/loso_pooled_predictions.csv"

meta = pd.read_csv(META)
pred = pd.read_csv(PRED)
child = pred.groupby("participant_id").agg(is_asd=("is_asd", "first"), p_asd=("p_asd", "mean")).reset_index()
child = child.merge(meta, left_on="participant_id", right_on="ParticipantID")
asd, td = child[child.is_asd == 1], child[child.is_asd == 0]
missed, caught = asd[asd.p_asd < 0.5], asd[asd.p_asd >= 0.5]

rho_cars = spearmanr(asd["CARS Score"], asd.p_asd)
rho_age_asd = spearmanr(asd.Age, asd.p_asd)
rho_age_td = spearmanr(td.Age, td.p_asd)
mw_cars = mannwhitneyu(missed["CARS Score"], caught["CARS Score"])
mw_age = mannwhitneyu(missed.Age, caught.Age)

lines = [
    "# LOSO errors vs clinical variables",
    "",
    f"{len(asd)} ASD / {len(td)} TD children; child score = mean P(ASD); missed ASD = score < 0.5 ({len(missed)}).",
    "Exploratory, 5 tests, no multiplicity correction.",
    "",
    "| Test | Group | Statistic | p |",
    "|---|---|---|---|",
    f"| Spearman CARS vs P(ASD) | ASD (n={len(asd)}) | ρ = {rho_cars.statistic:.3f} | {rho_cars.pvalue:.4f} |",
    f"| Spearman age vs P(ASD) | ASD (n={len(asd)}) | ρ = {rho_age_asd.statistic:.3f} | {rho_age_asd.pvalue:.4f} |",
    f"| Spearman age vs P(ASD) | TD (n={len(td)}) | ρ = {rho_age_td.statistic:.3f} | {rho_age_td.pvalue:.4f} |",
    f"| Mann-Whitney CARS, missed vs detected | ASD ({len(missed)} vs {len(caught)}) | U = {mw_cars.statistic:.1f} | {mw_cars.pvalue:.4f} |",
    f"| Mann-Whitney age, missed vs detected | ASD ({len(missed)} vs {len(caught)}) | U = {mw_age.statistic:.1f} | {mw_age.pvalue:.4f} |",
    "",
    f"Missed ASD: CARS {missed['CARS Score'].tolist()}, age {missed.Age.tolist()}; "
    f"detected ASD median CARS {caught['CARS Score'].median()}, median age {caught.Age.median()}.",
]
with open("artifacts/loso/loso_clinical.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
