import streamlit as st
import pandas as pd

def render_scanner_table(scan_log, current_step_idx=None):
    if not scan_log:
        st.info("No scan data available.")
        return

    df = pd.DataFrame(scan_log)

    def highlight_current(row):
        if current_step_idx is not None and row.name == current_step_idx:
            return ['background-color: #f59e0b; color: black'] * len(row)
        return [''] * len(row)

    st.dataframe(
        df.style.apply(highlight_current, axis=1).format({
            "Interval Start": "{:.4f}", "Interval End": "{:.4f}",
            "f(Start)": "{:.4f}", "f(End)": "{:.4f}"
        }),
        use_container_width=True,
        height=400,
        hide_index=True  # <--- Hides index here too
    )

def render_iteration_table(history, current_step_idx=None, tol=0.001):
    if not history:
        st.info("No iteration data available.")
        return

    df = pd.DataFrame(history)

    # 1. Styling Logic
    def highlight_row(row):
        styles = [''] * len(row)
        
        # --- A. RETRIEVE VALUES SAFELY ---
        try:
            err_val = float(row.get('Error % (Ea)', 100))
        except: err_val = 100.0
        
        # Check both f(c) (Bisection) and f(x) (Newton/Secant)
        f_val = 100.0
        if 'f(c)' in row: f_val = float(row['f(c)'])
        elif 'f(x)' in row: f_val = float(row['f(x)'])
        
        is_converged = (err_val < tol) and (abs(f_val) < 0.1)

        if is_converged:
             return ['background-color: #dcfce7; color: #14532d; font-weight: bold'] * len(row)
        
        # --- C. ACTIVE STEP HIGHLIGHT ---
        # Adjust for 0-based index vs 1-based Iter
        if current_step_idx is not None:
             # If explicit 'Iter' column exists, use it, otherwise use index
             iter_num = int(row.get('Iter', row.name + 1))
             if iter_num == current_step_idx:
                return ['background-color: #fff9c4; color: black'] * len(row)

        return styles

    # 2. Render with Index Hidden
    st.dataframe(
        df.style.apply(highlight_row, axis=1).format(
            lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else str(x)
        ),
        use_container_width=True,
        height=450,
        hide_index=True # <--- REMOVES THE INDEX NUMBER COLUMN
    )