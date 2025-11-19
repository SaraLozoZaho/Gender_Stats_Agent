# GENDER_STATS_AGENT/gui/interface.py

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd

from agent.agent_core import run_gender_expert_agent
from analysis.statistics_pipeline_gender import build_gender_stats_payload


# ============================================================
# Format stats nicely for display
# ============================================================

def format_stats_for_display(stats_json: dict) -> str:
    lines = []

    info = stats_json.get("dataset_info", {})
    lines.append("=== DATASET INFORMATION ===")
    lines.append(f"Rows: {info.get('n_rows')}")
    lines.append(f"Columns: {info.get('n_cols')}")
    lines.append("")

    # SEX COUNTS
    lines.append("=== SEX DISTRIBUTION (1=Male, 2=Female) ===")
    sex_counts = info.get("count_by_sex", {})
    for s, n in sex_counts.items():
        lines.append(f"  Sex {s}: n={n}")
    lines.append("")

    # DESCRIPTIVES
    desc = stats_json.get("descriptive", {})
    lines.append("=== DESCRIPTIVE STATISTICS ===")

    if "by_sex" in desc:
        for sex, vars_ in desc["by_sex"].items():
            lines.append(f"\nSex = {sex}:")
            for var, st in vars_.items():
                mean = st.get("mean")
                if mean is not None:
                    lines.append(f"  {var}: mean = {mean:.3f}")
    lines.append("")

    # CORRELATIONS
    corr = stats_json.get("correlations", {})
    lines.append("=== STRONG CORRELATIONS (|r| ≥ 0.7) ===")
    strong = corr.get("strong_pairs", [])
    if strong:
        for pair in strong:
            lines.append(f"  {pair['var1']} ↔ {pair['var2']} (r = {pair['r']})")
    else:
        lines.append("  None found.")
    lines.append("")

    # ANOVA
    anova = stats_json.get("anova_sex", {})
    lines.append("=== ANOVA BY SEX ===")
    for var, res in anova.items():
        if isinstance(res, dict) and res.get("p") is not None:
            lines.append(f"  {var}: F={res.get('F')}, p={res.get('p')}")
    lines.append("")

    # REGRESSIONS
    regs = stats_json.get("regressions", {})
    lines.append("=== REGRESSIONS ===")
    for name, reg in regs.items():
        if "coef" in reg:
            lines.append(
                f"  {name}: coef={reg['coef']:.3f}, R²={reg['r2']:.3f}, p={reg['p']:.3e}"
            )

    return "\n".join(lines)


# ============================================================
# GUI APPLICATION
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
        values=["investigator", "clinical", "short_clinical", "reviewer", "grant"],
        state="readonly",
        width=25,
    )
    style_combo.pack(side="left", padx=10)

    # QUESTION BOX
    tk.Label(top_frame, text="Ask the expert:").pack(side="left", padx=(20, 5))
    question_box = tk.Text(top_frame, height=3, width=55)
    question_box.pack(side="left", padx=10)

    # STATE
    stats_state = {"stats": None}
    conversation_state = {"history": []}

    # --------------------------------------------
    # LOAD DATA
    # --------------------------------------------
    def load_and_analyze():
        path = filedialog.askopenfilename(
            initialdir=r"Y:\BWSync\Agents\Gender_Stats_Agent\data",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")],
        )
        if not path:
            return

        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        except Exception as e:
            messagebox.showerror("Error loading file", f"{e}")
            return

        if "Sex" not in df.columns:
            messagebox.showwarning("Missing 'Sex' column", "Dataset must have a column named 'Sex'.")
            return

        try:
            stats_json = build_gender_stats_payload(df, sex_col="Sex")
        except Exception as e:
            messagebox.showerror("Analysis error", f"{e}")
            return

        stats_state["stats"] = stats_json

        text = format_stats_for_display(stats_json)
        stats_text_box.config(state="normal")
        stats_text_box.delete("1.0", tk.END)
        stats_text_box.insert(tk.END, text)
        stats_text_box.config(state="disabled")

        messagebox.showinfo("Done", "Statistical analysis completed.")

    # --------------------------------------------
    # ASK EXPERT AGENT
    # --------------------------------------------
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

    # SAVE REPORT
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
