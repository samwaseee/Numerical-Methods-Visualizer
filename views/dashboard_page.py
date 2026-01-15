import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from components.header import render_function_header
from components.math_display import display_math_analysis
from components.graph_plotter import render_scanner_chart, render_iteration_chart
from components.results_table import render_scanner_table, render_iteration_table

def trigger_back_to_results():
    """
    Handles navigation back from Dashboard.
    Forces a valid state to prevent blank URL params.
    """
    if st.session_state.params.get('method') == "Find All Roots (Scan)":
        # --- FIX: HARD RESET TO DEFAULTS ---
        # If we don't do this, we get "func=&range_a=0" which breaks the UI.
        st.session_state.page = "input"
        
        default_func = "x^4 + 6*x^3 - 77*x^2 - 330*x + 400"
        
        # 1. Update Session State
        st.session_state.func_str = default_func
        st.session_state.g_str = f"x - ({default_func})/1000"
        st.session_state.range_a = -10.0
        st.session_state.range_b = 10.0
        st.session_state.scanner_step = 0.15
        
        # 2. Update URL (The "Source of Truth")
        st.query_params["page"] = "input"
        st.query_params["func"] = default_func
        st.query_params["g"] = st.session_state.g_str
        st.query_params["a"] = "-10.0"
        st.query_params["b"] = "10.0"
        st.query_params["step"] = "0.15"
        # -----------------------------------
        
    else:
        # For single root methods, just go back to results
        st.session_state.page = "result"
        st.query_params["page"] = "result"

def show_dashboard_page(header_container):
    solver = st.session_state.solver
    p = st.session_state.params
    
    with header_container.container():
        render_function_header(solver)
    
    # Back Button
    if st.button("Back", icon="⬅️", on_click=trigger_back_to_results): pass
    
    st.title(f"{p['method']}")

    history = []
    roots_list = []
    scan_log = []
    
    # 1. SAFETY BRAKE
    if p['method'] == "Find All Roots (Scan)":
        total_range = abs(p['b'] - p['a'])
        step_size = p['step']
        if step_size <= 0:
            st.error("Step size must be positive!", icon="🚫")
            return
        expected_steps = total_range / step_size
        if expected_steps > 5000:
            st.error(f"⚠️ **Memory Overload Prevented**", icon="🛑")
            st.markdown(f"Reduce range or increase step size. Steps: {int(expected_steps):,}")
            st.stop()

    # 2. EXECUTE ALGORITHMS
    try:
        if p['method'] == "Find All Roots (Scan)":
            roots_list, scan_log = solver.find_all_roots(p['a'], p['b'], p['step'])
        elif p['method'] == "Bisection Method":
            history = solver.bisection_method(p['a'], p['b'], p['iters'])
        elif p['method'] == "False Position":
            history = solver.false_position_method(p['a'], p['b'], p['iters'])
        elif p['method'] == "Fixed Point Iteration":
            res = solver.fixed_point_method(p['g_str'], p['x0'], p['iters'])
            if res: history, _ = res
        elif p['method'] == "Newton-Raphson":
            history = solver.newton_raphson_method(p['x0'], p['iters'])
        elif p['method'] == "Secant Method":
            history = solver.secant_method(p['x0'], p.get('x1', p['x0']+1), p['iters'])
    except Exception as e: st.error(f"Computation Error: {e}", icon="🚨")

    # --- VIEW: SCANNER ---
    if p['method'] == "Find All Roots (Scan)":
        with st.container(border=True):
            col_res, col_advice = st.columns([1, 2])
            with col_res: st.metric("Roots Found", len(roots_list))
            with col_advice:
                try:
                    s_clean = solver.original_str.replace("^", "**")
                    x_sym = sp.symbols('x')
                    expr = sp.sympify(s_clean)
                    if expr.is_polynomial(x_sym):
                        degree = sp.degree(expr, x_sym)
                        if len(roots_list) < degree:
                            st.warning(f"Found {len(roots_list)}/{degree} roots.", icon="⚠️")
                        else:
                            st.success("All roots found.", icon="✅")
                except: pass

        # 1. Chart
        current_scan_row, step_idx = render_scanner_chart(solver, p['a'], p['b'], p['step'], roots_list, scan_log)
        
        st.markdown("---")
        
        # 2. Detailed Table
        st.subheader("📊 Scan Details")
        st.caption("Why does an interval have a root? Look for **YES** in the 'Sign Change' column ($f(a) \cdot f(b) < 0$).")
        
        if scan_log:
            df_log = pd.DataFrame(scan_log)
            def highlight_root(row):
                if "YES" in str(row['Sign Change?']):
                    return ['background-color: #1f3a2d'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_log.style.apply(highlight_root, axis=1).format({
                    "Interval Start": "{:.4f}",
                    "Interval End": "{:.4f}",
                    "f(Start)": "{:.4f}",
                    "f(End)": "{:.4f}"
                }),
                use_container_width=True,
                height=400,
                column_config={
                    "Sign Change?": st.column_config.TextColumn("Sign Change?")
                }
            )

        st.markdown("---")
        if current_scan_row is not None:
            display_math_analysis(p['method'], current_scan_row, solver, extra_params={'step': p['step']})

    # --- VIEW: ITERATIVE METHODS ---
    elif history:
        st.subheader("Iteration Playback")
        col_viz, col_log = st.columns([1.5, 1])
        
        with col_viz:
            row_data, step_idx = render_iteration_chart(solver, history, p['method'], p.get('g_str'))
        with col_log:
            render_iteration_table(history, step_idx, tol=p.get('tol', 0.001))

        st.markdown("---")
        display_math_analysis(p['method'], row_data, solver)