# GENDER_STATS_AGENT/gui/interface.py

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd

from agent.agent_core import run_gender_expert_agent
from gender_agent_core import build_gender_stats_payload   # ← Debe existir


# ============================================================
# Helper to format the gender statistics for display
# ============================================================

def format_stats_for_display(stats_json: dict) -> str:
    lines = []

    # === DATASET INFO ===
    info = stats_json.get("dataset_info", {})
    lines.append("=== DATASET INFORMATION ===")
    lines.append(f"Total rows: {info.get('n_total')}")
    lines.append("")

    # === SEX/GENDER COUNTS ===
    sex_counts = info.get("count_by_sex", {})
    gender_counts = info.get("count_by_gender", {})

    lines.append("=== SEX / GENDER DISTRIBUTION ===")
    if sex_counts:
        lines.append("By sex:")
        for s, n in sex_counts.items():
            lines.append(f"  {s}: {n}")
    if gender_counts:
        lines.append("\nBy gender:")
        for s, n in gender_counts.items():
            lines.append(f"  {s}: {n}")
    lines.append("")

    # === DESCRIPTIVES ===
    desc = stats_json.get("descriptive", {})
    lines.append("=== DESCRIPTIVE STATISTICS ===")
    if "by_sex" in desc:
        lines.append("By sex:")
        for grp, vars_ in desc["by_sex"].items():
            lines.append(f"\n  Sex = {grp}")
            for var, st in vars_.items():
                if "mean" in st:
                    lines.append(f"    {var}: mean={st['mean']:.3f}")
    lines.append("")

    # === GROUP TESTS ===
    tests = stats_json.get("group_tests", {})
    lines.append("=== GROUP COMPARISONS ===")
    for var, res in tests.items():
        p = res.get("p")
        effect = res.get("effect")
        test = res.get("test_name")
        lines.append(f"  {var}: {test}, effect={effect}, p={p}")
    lines.append("")

    # === MODELS ===
    models = stats_json.get("models", {})
    lines.append("=== PREDICTIVE MODELS ===")
    for name, res in models.items():
        metric = res.get("metric")
        lines.append(f"  {name}: performance metric = {metric}")
    lines.append("")

    return "\n".join(lines)



# ============================================================
# MAIN GUI APPLICATION
# ============================================================

def launch_gui():
    root = tk.Tk()
    root.title("Gender / Sex Health Analysis Agent")
    root.geometry("1300x900")

    # ---------------- TOP PANEL ----------------
    top_frame = tk.Frame(root)
    top_frame.pack(fill="x", padx=10, pady=5)

    # Dropdown for style
    tk.Label(top_frame, text="Report style:").pack(side="left", padx=(0, 5))
    report_style_var = tk.StringVar(value="investigator")
    style_combo = ttk.Combobox(
        top_frame,
        textvariable=report_style_var,
        values=[
            "investigator",
            "clinical",
            "short_clinical",
            "reviewer",
            "grant",
        ],
        state="readonly",
        width=25,
    )
    style_combo.pack(side="left", padx=10)

    # Question box
    tk.Label(top_frame, text="Ask the expert:").pack(side="left", padx=(20, 5))
    question_box = tk.Text(top_frame, height=3, width=55)
    question_box.pack(side="left", padx=10)

    # State holders
    stats_state = {"stats": None}
    conversation_state = {"history": []}

    # ---------------- LOAD & ANALYZE ----------------
    def load_and_analyze():
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")]
        )
        if not path:
            return

        try:
            df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        except Exception as e:
            messagebox.showerror("Loading Error", str(e))
            return

        try:
            stats_json = build_gender_stats_payload(df)
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            return

        stats_state["stats"] = stats_json

        # Show stats
        text = format_stats_for_display(stats_json)
        stats_text_box.config(state="normal")
        stats_text_box.delete("1.0", tk.END)
        stats_text_box.insert(tk.END, text)
        stats_text_box.config(state="disabled")

        messagebox.showinfo("Completed", "Statistical analysis done.")

    # ---------------- ASK EXPERT ----------------
    def run_agent_call():
        stats_json = stats_state.get("stats")
        if stats_json is None:
            messagebox.showwarning("No Data", "Load dataset first.")
            return

        style = report_style_var.get()
        question = question_box.get("1.0", tk.END).strip()

        try:
            result = run_gender_expert_agent(
                stats_json=stats_json,
                report_style=style,
                question=question,
                history=conversation_state["history"]
            )
        except Exception as e:
            messagebox.showerror("Agent Error", str(e))
            return

        # Save to history
        conversation_state["history"].append({"role": "user", "content": question})
        conversation_state["history"].append({"role": "assistant", "content": result})

        # Display
        interp_text_box.config(state="normal")
        interp_text_box.insert(tk.END, f"\nUSER:\n{question}\n\n")
        interp_text_box.insert(tk.END, f"ASSISTANT ({style}):\n{result}\n\n")
        interp_text_box.config(state="disabled")

        question_box.delete("1.0", tk.END)
        notebook.select(0)

    # ---------------- RESET ----------------
    def reset_conversation():
        conversation_state["history"] = []
        interp_text_box.config(state="normal")
        interp_text_box.delete("1.0", tk.END)
        interp_text_box.config(state="disabled")
        question_box.delete("1.0", tk.END)
        messagebox.showinfo("Reset", "Conversation cleared.")

    # ---------------- SAVE REPORT ----------------
    def save_report():
        content = interp_text_box.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Empty", "No report to save.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        messagebox.showinfo("Saved", f"Saved to: {path}")

    # Buttons
    tk.Button(top_frame, text="1) Load & Analyze", command=load_and_analyze).pack(side="left", padx=10)
    tk.Button(top_frame, text="2) Ask Expert Agent", command=run_agent_call).pack(side="left", padx=10)
    tk.Button(top_frame, text="Reset", command=reset_conversation).pack(side="left", padx=10)
    tk.Button(top_frame, text="Save Report", command=save_report).pack(side="left", padx=10)

    # Notebook tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    # Tab: Interpretation
    tab_interp = tk.Frame(notebook)
    notebook.add(tab_interp, text="Expert Interpretation")
    interp_text_box = tk.Text(tab_interp, wrap="word")
    interp_text_box.pack(fill="both", expand=True)

    # Tab: Stats
    tab_stats = tk.Frame(notebook)
    notebook.add(tab_stats, text="Statistics")
    stats_text_box = tk.Text(tab_stats, wrap="word", state="disabled")
    stats_text_box.pack(fill="both", expand=True)

    root.mainloop()
