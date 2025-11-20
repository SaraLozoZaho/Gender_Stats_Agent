# GENDER_STATS_AGENT/gui/interface.py

import tkinter as tk
import traceback
import sys
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd

from agent.agent_core import run_gender_expert_agent
from analysis.statistics_pipeline_gender import build_gender_stats_payload


# ============================================================
# PRINT FULL TRACEBACK TO CONSOLE
# ============================================================

def print_exception_to_console(e):
    print("\n" + "="*80)
    print("GUI ERROR:")
    traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
    print("="*80 + "\n")


# ============================================================
# FORMAT STATS (IMPROVED)
# ============================================================

def format_stats_for_display(stats_json: dict) -> str:
    lines = []

    # DATASET INFO
    info = stats_json.get("dataset_info", {})
    lines.append("========================================")
    lines.append("           DATASET INFORMATION")
    lines.append("========================================")
    lines.append(f"Rows:     {info.get('n_rows')}")
    lines.append(f"Columns:  {info.get('n_cols')}")
    lines.append("")

    # SEX COUNTS
    lines.append("=== SEX DISTRIBUTION (1=Male, 2=Female) ===")
    for sex, n in info.get("count_by_sex", {}).items():
        lines.append(f"  Sex {sex}: n={n}")
    lines.append("")

    # DESCRIPTIVES
    lines.append("========================================")
    lines.append("          DESCRIPTIVE STATISTICS")
    lines.append("========================================")
    desc = stats_json.get("descriptive", {})
    for sex, vars_ in desc.get("by_sex", {}).items():
        lines.append(f"\nSex = {sex}:")
        for var, st in vars_.items():
            mean = st.get("mean")
            sd = st.get("sd")
            if mean is not None:
                lines.append(f"  {var}: mean={mean:.3f}, sd={sd}")
    lines.append("")

    # CORRELATIONS
    lines.append("========================================")
    lines.append("          STRONG CORRELATIONS")
    lines.append("========================================")
    strong = stats_json.get("correlations", {}).get("strong_pairs", [])
    if strong:
        for p in strong:
            lines.append(f"  {p['var1']} ↔ {p['var2']} (r={p['r']:.3f})")
    else:
        lines.append("  None found.")
    lines.append("")

    # ANOVA
    lines.append("========================================")
    lines.append("               ANOVA BY SEX")
    lines.append("========================================")

    anova = stats_json.get("anova_sex", {})

    for var, res in anova.items():
        if not isinstance(res, dict):
            continue

        F = res.get("F")
        p = res.get("p")
        gs = res.get("group_sizes", {})

        n1 = gs.get("1")
        n2 = gs.get("2")

        if F is None or p is None:
            reason = res.get("reason", "ANOVA could not be computed.")
            lines.append(f"{var}: {reason} (n1={n1}, n2={n2})")
        else:
            lines.append(f"{var}: F={F:.3f}, p={p:.3e} (n1={n1}, n2={n2})")


    lines.append("")

    # T-TESTS / U-TESTS
    lines.append("========================================")
    lines.append("       GROUP TESTS (t-test / U-test)")
    lines.append("========================================")
    for var, res in stats_json.get("t_tests", {}).items():
        lines.append(f"\n{var}:")
        lines.append(f"  Test:       {res['test']}")
        lines.append(f"  Effect:     {res['effect']:.3f}")
        lines.append(f"  p-value:    {res['p']:.3e}")
        if res["cohen_d"] is not None:
            lines.append(f"  Cohen's d:  {res['cohen_d']:.3f}")
        ci = res["ci"]
        if ci:
            lines.append(f"  95% CI:     [{ci[0]:.3f}, {ci[1]:.3f}]")
        gs = res["group_sizes"]
        # Convert keys to sorted order for consistent printing
        keys = sorted(gs.keys(), key=lambda x: str(x))

        for k in keys:
            lines.append(f"  n (sex={k}) = {gs[k]}")

    lines.append("")

    # CHI-SQUARE
    lines.append("========================================")
    lines.append("      CHI-SQUARE TESTS (categorical)")
    lines.append("========================================")
    chi = stats_json.get("categorical_tests", {})
    if chi:
        for col, res in chi.items():
            lines.append(f"\n{col}:")
            lines.append(f"  χ² = {res['chi2']:.3f}")
            lines.append(f"  p  = {res['p']:.3e}")
    else:
        lines.append("  No categorical tests available.")
    lines.append("")

    # LOGISTIC REGRESSION
    lines.append("========================================")
    lines.append("       LOGISTIC REGRESSION")
    lines.append("========================================")
    logreg = stats_json.get("logistic_regression", {})
    if logreg:
        if "error" in logreg:
            lines.append("  ERROR in logistic regression:")
            lines.append(f"  {logreg['error']}")
        else:
            lines.append(f"Outcome:      {logreg['outcome']}")
            lines.append(f"Odds Ratio:   {logreg['OR']:.3f}")
            lines.append(f"p-value:      {logreg['p']:.3e}")
            lines.append(f"n:            {logreg['n']}")
    else:
        lines.append("  No binary outcome detected.")
    lines.append("")

    # SUMMARY
    lines.append("========================================")
    lines.append("                SUMMARY")
    lines.append("========================================")
    summary = stats_json.get("gender_gap_indicators", {})
    lines.append(f"Significant t-tests:     {summary.get('significant_t_tests', 0)}")
    lines.append(f"Significant ANOVA:       {summary.get('significant_anova', 0)}")
    lines.append(f"Significant categorical: {summary.get('significant_categorical', 0)}")

    return "\n".join(lines)


# ============================================================
# GUI
# ============================================================

def launch_gui():
    root = tk.Tk()
    root.title("Gender / Sex Health Analysis Agent")
    root.geometry("1300x900")

    # TOP PANEL
    top_frame = tk.Frame(root)
    top_frame.pack(fill="x", padx=10, pady=5)

    tk.Label(top_frame, text="Report style:").pack(side="left")
    report_style_var = tk.StringVar(value="investigator")

    style_combo = ttk.Combobox(
        top_frame,
        textvariable=report_style_var,
        values=[
            "investigator",
            "investigator_advanced",
            "clinical",
            "clinical_advanced",
            "short_clinical",
            "reviewer",
            "grant",
        ],
        state="readonly",
        width=25,
    )
    style_combo.pack(side="left", padx=10)

    # QUESTION
    tk.Label(top_frame, text="Ask the expert:").pack(side="left", padx=(20, 5))
    question_box = tk.Text(top_frame, height=3, width=55)
    question_box.pack(side="left", padx=10)

    # STATE
    stats_state = {"stats": None}
    conversation_state = {"history": []}

    # LOAD DATA
    def load_and_analyze():
        path = filedialog.askopenfilename(
            initialdir=r"Y:\BWSync\Agents\Gender_Stats_Agent\data",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")],
        )
        if not path:
            return

        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)

            # Limpieza global para evitar errores silenciosos
            df.columns = (
                df.columns
                .astype(str)
                .str.replace("\ufeff", "", regex=False)   # elimina BOM
                .str.replace("\xa0", " ", regex=False)    # elimina espacio NO separable
                .str.strip()                              # quita espacios normales
            )

            # eliminar duplicadas después de limpiar completamente
            df = df.loc[:, ~df.columns.duplicated()]


            print("\n=== COLUMNAS DEL DATAFRAME ===")
            print(df.columns.tolist())
            print("===============================\n")

            stats_json = build_gender_stats_payload(df, sex_col="Sex")

        except Exception as e:
            print_exception_to_console(e)
            messagebox.showerror("Error loading file", f"{e}")
            return

        if "Sex" not in df.columns:
            messagebox.showwarning("Missing 'Sex' column", "Dataset must have a column 'Sex'.")
            return

        try:
            stats_json = build_gender_stats_payload(df, sex_col="Sex")
        except Exception as e:
            print_exception_to_console(e)
            messagebox.showerror("Analysis error", f"{e}")
            return

        stats_state["stats"] = stats_json

        text = format_stats_for_display(stats_json)
        stats_text_box.config(state="normal")
        stats_text_box.delete("1.0", tk.END)
        stats_text_box.insert(tk.END, text)
        stats_text_box.config(state="disabled")

        messagebox.showinfo("Done", "Statistical analysis completed.")

    # RUN AGENT
    def run_agent_call():
        stats_json = stats_state.get("stats")
        if stats_json is None:
            messagebox.showwarning("No data", "Load a dataset first.")
            return

        style = report_style_var.get()
        question = question_box.get("1.0", tk.END).strip()

        try:
            result = run_gender_expert_agent(
                stats_json=stats_json,
                report_style=style,
                question=question,
                history=conversation_state["history"],
            )
        except Exception as e:
            print_exception_to_console(e)
            messagebox.showerror("Agent error", f"{e}")
            return

        conversation_state["history"].append({"role": "user", "content": question})
        conversation_state["history"].append({"role": "assistant", "content": result})

        interp_text_box.config(state="normal")
        interp_text_box.insert(tk.END, f"\nUSER:\n{question}\n\n")
        interp_text_box.insert(tk.END, f"ASSISTANT ({style}):\n{result}\n\n")
        interp_text_box.config(state="disabled")

        question_box.delete("1.0", tk.END)
        notebook.select(0)

    # RESET
    def reset_conversation():
        conversation_state["history"] = []
        interp_text_box.config(state="normal")
        interp_text_box.delete("1.0", tk.END)
        interp_text_box.config(state="disabled")
        question_box.delete("1.0", tk.END)

        messagebox.showinfo("Reset", "Conversation cleared.")

    # SAVE
    def save_report():
        content = interp_text_box.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Empty", "No report to save.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")]
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        messagebox.showinfo("Saved", f"Saved to: {path}")

    # BUTTONS
    tk.Button(top_frame, text="1) Load & Analyze", command=load_and_analyze).pack(side="left", padx=10)
    tk.Button(top_frame, text="2) Ask Expert Agent", command=run_agent_call).pack(side="left", padx=10)
    tk.Button(top_frame, text="Reset", command=reset_conversation).pack(side="left", padx=10)
    tk.Button(top_frame, text="Save Report", command=save_report).pack(side="left", padx=10)

    # NOTEBOOK
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    # Interpretation tab
    tab_interp = tk.Frame(notebook)
    notebook.add(tab_interp, text="Expert Interpretation")
    interp_text_box = tk.Text(tab_interp, wrap="word")
    interp_text_box.pack(fill="both", expand=True)

    # Stats tab
    tab_stats = tk.Frame(notebook)
    notebook.add(tab_stats, text="Statistics")
    stats_text_box = tk.Text(tab_stats, wrap="word", state="disabled")
    stats_text_box.pack(fill="both", expand=True)

    root.mainloop()
