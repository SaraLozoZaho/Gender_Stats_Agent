# GENDER_STATS_AGENT/agent/agent_core.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# 1. COMPRESS GENDER STATS (safe, anonymous summary)
# ============================================================

def compress_gender_stats(stats_json: dict) -> str:
    """
    Summarize gender/sex-related statistics safely without exposing raw data.
    """
    lines = []

    # === DATASET INFO ===
    info = stats_json.get("dataset_info", {})
    n_total = info.get("n_total")
    counts_sex = info.get("count_by_sex", {})
    counts_gender = info.get("count_by_gender", {})

    lines.append("DATASET OVERVIEW:")
    if n_total is not None:
        lines.append(f"- Total sample size: n={n_total}")
    if counts_sex:
        lines.append("- Counts by sex:")
        for k, v in counts_sex.items():
            lines.append(f"    {k}: n={v}")
    if counts_gender:
        lines.append("- Counts by gender:")
        for k, v in counts_gender.items():
            lines.append(f"    {k}: n={v}")
    if not (n_total or counts_sex or counts_gender):
        lines.append("- No dataset info available.")

    # === DESCRIPTIVES BY SEX ===
    desc = stats_json.get("descriptive", {})
    by_sex = desc.get("by_sex", {})
    lines.append("\nDESCRIPTIVE STATISTICS BY SEX:")
    if by_sex:
        for sex, vars_ in by_sex.items():
            lines.append(f"- Sex = {sex}:")
            for var, st in vars_.items():
                mean = st.get("mean")
                sd = st.get("sd")
                if mean is not None:
                    s = f"    {var}: mean={mean:.3f}"
                    if sd is not None:
                        s += f", sd={sd:.3f}"
                    lines.append(s)
    else:
        lines.append("- No sex-stratified descriptives available.")

    # === DESCRIPTIVES BY GENDER ===
    by_gender = desc.get("by_gender", {})
    lines.append("\nDESCRIPTIVE STATISTICS BY GENDER:")
    if by_gender:
        for g, vars_ in by_gender.items():
            lines.append(f"- Gender = {g}:")
            for var, st in vars_.items():
                mean = st.get("mean")
                sd = st.get("sd")
                if mean is not None:
                    s = f"    {var}: mean={mean:.3f}"
                    if sd is not None:
                        s += f", sd={sd:.3f}"
                    lines.append(s)
    else:
        lines.append("- No gender-stratified descriptives available.")

    # === GROUP COMPARISONS ===
    comps = stats_json.get("group_tests", {})
    lines.append("\nGROUP COMPARISONS:")
    if comps:
        for var, res in comps.items():
            effect = res.get("effect")
            ci = res.get("ci")
            p = res.get("p")
            test = res.get("test_name")
            note = res.get("note")

            line = f"- {var}:"
            if test:
                line += f" test={test}"
            if effect is not None:
                line += f", effect={effect:.3f}"
            if ci and len(ci) == 2:
                line += f", CI=({ci[0]:.3f}, {ci[1]:.3f})"
            if p is not None:
                line += f", p={p:.3e}"
            if note:
                line += f" [{note}]"

            lines.append(line)
    else:
        lines.append("- No group test results found.")

    # === PREDICTIVE MODELS ===
    models = stats_json.get("models", {})
    lines.append("\nPREDICTIVE MODELS INCLUDING SEX/GENDER:")
    if models:
        for name, res in models.items():
            metric = res.get("metric")
            sex_effect = res.get("sex_effect")
            gender_effect = res.get("gender_effect")

            lines.append(f"- Model: {name}")
            if metric is not None:
                lines.append(f"    Performance metric: {metric}")
            if sex_effect:
                lines.append(f"    Sex effect: {sex_effect}")
            if gender_effect:
                lines.append(f"    Gender effect: {gender_effect}")
            if res.get("notes"):
                lines.append(f"    Notes: {res['notes']}")
    else:
        lines.append("- No predictive models computed.")

    # === GENDER/SEX GAP INDICATORS ===
    gap = stats_json.get("gender_gap_indicators", {})
    lines.append("\nGENDER / SEX GAP INDICATORS:")
    if gap:
        for item, val in gap.items():
            lines.append(f"- {item}: {val}")
    else:
        lines.append("- No gender-gap indicators provided.")

    return "\n".join(lines)


# ============================================================
# 2. EXPERT INTERPRETATION AGENT (OpenAI)
# ============================================================

def run_gender_expert_agent(
    stats_json: dict,
    question: str | None = None,
    style: str = "investigator",
    report_style: str | None = None,
    history: list | None = None,
):
    """
    Expert agent for interpreting sex/gender-related medical data.
    Only aggregated statistics (no raw data) must be passed to this function.
    """

    if client.api_key is None or client.api_key.strip() == "":
        raise RuntimeError("OPENAI_API_KEY missing. Cannot call the model.")

    # ---- SELECT STYLE ----
    if report_style is not None:
        style = report_style

    style_map = {
        "clinical": (
            "Provide a clinical interpretation for anesthesiology and perioperative medicine, "
            "focusing on sex/gender risk differences and implications for patient care."
        ),
        "short_clinical": (
            "Provide short bullet points (max 10) with the clinically most relevant sex/gender differences."
        ),
        "investigator": (
            "Write as a senior researcher and biostatistician. Discuss mechanisms, confounding, "
            "effect modification, limitations, and implications for future research."
        ),
        "reviewer": (
            "Write as a peer reviewer for a high-impact journal. Critically assess methods, "
            "risk of bias, adequacy of sex/gender analysis, and reporting quality."
        ),
        "grant": (
            "Write as a grant reviewer (DFG/NIH). Evaluate strengths and weaknesses of the "
            "sex/gender analysis and suggest improvements."
        ),
    }

    style_instruction = style_map.get(style, "Write a scientific sex/gender interpretation.")

    # ---- COMPRESS DATA ----
    stats_text = compress_gender_stats(stats_json)

    # ---- SYSTEM PROMPT ----
    system_prompt = f"""
You are an expert in biostatistics, anesthesiology, perioperative medicine, and sex/gender-sensitive research.
You are deeply familiar with gender medicine, gender bias, and statistical best practices.

STRICT RULES:
- Never invent numbers or sample sizes.
- Do NOT assume data that is not explicitly provided.
- Base everything ONLY on the aggregated dataset summary.
- If data are insufficient, clearly state the limitation.
- Emphasize effect sizes, clinical relevance, and methodological robustness.
- Follow the chosen style: {style_instruction}
"""

    # ---- USER PROMPT ----
    user_prompt = f"""
### DATA SUMMARY ###
{stats_text}

### USER QUESTION ###
{question if question else "Please provide a complete interpretation of sex/gender differences based on the results."}

Consider previous conversations if provided.
"""

    # ---- BUILD MESSAGE LIST ----
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for entry in history:
            if isinstance(entry, dict) and "role" in entry:
                messages.append(entry)

    messages.append({"role": "user", "content": user_prompt})

    # ---- CALL MODEL ----
    response = client.responses.create(
        model="gpt-4.1",
        input=messages,
    )

    # ---- OUTPUT ----
    return response.output_text
