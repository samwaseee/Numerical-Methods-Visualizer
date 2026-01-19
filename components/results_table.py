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
    
    # Slice dataframe to only show rows up to current_step_idx
    if current_step_idx is not None and current_step_idx > 0:
        df = df.iloc[:current_step_idx]

    # 1. Styling Logic (works on original numeric values)
    def highlight_row(row):
        styles = [''] * len(row)
        
        # --- A. RETRIEVE VALUES SAFELY ---
        try:
            err_val = float(row.get('Error % (Ea)', 100))
        except: err_val = 100.0
        
        # Check both f(c) (Bisection) and f(x) (Newton/Secant)
        f_val = 100.0
        if 'f(c)' in row:
            try:
                f_val = float(row['f(c)'])
            except:
                f_val = abs(row['f(c)']) if isinstance(row['f(c)'], complex) else 100.0
        elif 'f(x)' in row:
            try:
                f_val = float(row['f(x)'])
            except:
                f_val = abs(row['f(x)']) if isinstance(row['f(x)'], complex) else 100.0
        
        is_converged = (err_val < tol) and (abs(f_val) < 0.1)

        if is_converged:
             return ['background-color: #dcfce7; color: #14532d; font-weight: bold'] * len(row)
        
        # --- C. ACTIVE STEP HIGHLIGHT ---
        # Use row.name (dataframe index) directly, which matches step_idx
        if current_step_idx is not None and row.name == current_step_idx - 1:
            return ['background-color: #fff9c4; color: black'] * len(row)

        return styles

    # 2. Custom formatter for display
    def format_value(val):
        # Handle complex numbers
        if isinstance(val, complex):
            real_part = round(val.real, 4)
            imag_part = round(val.imag, 4)
            sign = "+" if imag_part >= 0 else "-"
            return f"{real_part:.4f} {sign} {abs(imag_part):.4f}i"
        # Handle regular numbers
        if isinstance(val, (int, float)):
            if val == 0:
                return ""
            if abs(val) < 0.0001:
                return f"{val:.4e}"  # Scientific notation e.g., 1.2345e-10
            return f"{val:.4f}"
        # Keep strings as-is
        return str(val)
    
    # 3. Apply styling to original dataframe
    styled = df.style.apply(highlight_row, axis=1)
    
    # 4. Format all numeric and complex columns for display
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32'] or \
           any(isinstance(x, complex) for x in df[col] if pd.notna(x)):
            styled = styled.format({col: format_value})
    
    # 5. Render styled dataframe
    st.dataframe(
        styled,
        use_container_width=True,
        height=450,
        hide_index=True
    )