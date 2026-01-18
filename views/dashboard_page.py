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
    if st.session_state.params.get('method') == "Find All Roots (Scan)":
        st.session_state.page = "input"
        default_func = "x^4 + 6*x^3 - 77*x^2 - 330*x + 400"
        st.session_state.func_str = default_func
        st.session_state.g_str = f"x - ({default_func})/1000"
        st.session_state.range_a = -10.0
        st.session_state.range_b = 10.0
        st.session_state.scanner_step = 0.15
        st.query_params["page"] = "input"
        st.query_params["func"] = default_func
        st.query_params["g"] = st.session_state.g_str
        st.query_params["a"] = "-10.0"
        st.query_params["b"] = "10.0"
        st.query_params["step"] = "0.15"
    else:
        st.session_state.page = "result"
        st.query_params["page"] = "result"

def render_comparison_chart(solver, hist_a, name_a, hist_b, name_b):
    """
    Special Plotter for Comparison Mode.
    Plots two convergence paths on one graph.
    """
    # Determine bounds
    all_x = []
    if hist_a: all_x.extend([r.get('x_new', r.get('c (root)')) for r in hist_a])
    if hist_b: all_x.extend([r.get('x_new', r.get('c (root)')) for r in hist_b])
    
    if not all_x: all_x = [-10, 10]
    min_x, max_x = min(all_x), max(all_x)
    span = max_x - min_x if max_x != min_x else 2.0
    plot_min, plot_max = min_x - span*0.2, max_x + span*0.2
    
    x_range = np.linspace(plot_min, plot_max, 400)
    try: y_range = solver.f(x_range)
    except: y_range = np.zeros_like(x_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=y_range, name='f(x)', line=dict(color="#c3c8f6", width=3)))
    fig.add_hline(y=0, line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_color="gray", opacity=0.5)

    # Plot Method A (Circles)
    if hist_a:
        path_x = [r.get('x_new', r.get('c (root)')) for r in hist_a]
        path_y = [solver.f(x) for x in path_x]
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y, 
            mode='lines+markers', name=name_a,
            line=dict(color="#FB002A", dash='dot'),
            marker=dict(symbol='circle', size=8)
        ))

    # Plot Method B (Squares)
    if hist_b:
        path_x = [r.get('x_new', r.get('c (root)')) for r in hist_b]
        path_y = [solver.f(x) for x in path_x]
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y, 
            mode='lines+markers', name=name_b,
            line=dict(color='#10B981', dash='dash'),
            marker=dict(symbol='square', size=8)
        ))

    fig.update_layout(
        title=f"Race: {name_a} vs {name_b}",
        xaxis_title="x", yaxis_title="f(x)",
        height=500, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def show_dashboard_page(header_container):
    solver = st.session_state.solver
    p = st.session_state.params
    
    with header_container.container():
        render_function_header(solver)
    
    if st.button("Back", icon="⬅️", on_click=trigger_back_to_results): pass
    
    # --- COMPARISON MODE LOGIC ---
    if p.get('method') == "Comparison Mode":
        st.title("⚔️ Method Comparison Race")
        
        # 1. Run Method A
        hist_a = []
        name_a = p['method_a']
        try:
            if name_a == "Newton-Raphson": hist_a = solver.newton_raphson_method(p['x0'], p['iters'])
            elif name_a == "Bisection Method": hist_a = solver.bisection_method(p['a'], p['b'], p['iters'])
            elif name_a == "False Position": hist_a = solver.false_position_method(p['a'], p['b'], p['iters'])
            elif name_a == "Secant Method": hist_a = solver.secant_method(p['x0'], p['x0']+1, p['iters'])
            elif name_a == "Fixed Point Iteration": 
                res = solver.fixed_point_method(p['g_str'], p['x0'], p['iters'])
                if res: hist_a, _ = res
        except: pass

        # 2. Run Method B
        hist_b = []
        name_b = p['method_b']
        try:
            if name_b == "Newton-Raphson": hist_b = solver.newton_raphson_method(p['x0'], p['iters'])
            elif name_b == "Bisection Method": hist_b = solver.bisection_method(p['a'], p['b'], p['iters'])
            elif name_b == "False Position": hist_b = solver.false_position_method(p['a'], p['b'], p['iters'])
            elif name_b == "Secant Method": hist_b = solver.secant_method(p['x0'], p['x0']+1, p['iters'])
            elif name_b == "Fixed Point Iteration": 
                res = solver.fixed_point_method(p['g_str'], p['x0'], p['iters'])
                if res: hist_b, _ = res
        except: pass

        # 3. Render Joint Graph
        render_comparison_chart(solver, hist_a, name_a, hist_b, name_b)

        # 4. Render Stats Table
        st.subheader("📊 Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{name_a}**")
            if hist_a:
                last_a = hist_a[-1]
                st.metric("Iterations", len(hist_a))
                st.metric("Final Error", f"{last_a.get('Error % (Ea)', 0):.6f}%")
                st.metric("Final Root", f"{last_a.get('x_new', last_a.get('c (root)')):.6f}")
            else: st.error("Failed")
            
        with col2:
            st.markdown(f"**{name_b}**")
            if hist_b:
                last_b = hist_b[-1]
                st.metric("Iterations", len(hist_b))
                st.metric("Final Error", f"{last_b.get('Error % (Ea)', 0):.6f}%")
                st.metric("Final Root", f"{last_b.get('x_new', last_b.get('c (root)')):.6f}")
            else: st.error("Failed")

        return
    # -----------------------------

    st.title(f"{p['method']}")

    history = []
    roots_list = []
    scan_log = []
    
    if p['method'] == "Find All Roots (Scan)":
        total_range = abs(p['b'] - p['a'])
        step_size = p['step']
        if step_size <= 0:
            st.error("Step size must be positive!", icon="🚫")
            return
        expected_steps = total_range / step_size
        if expected_steps > 5000:
            st.error(f"⚠️ **Memory Overload Prevented**", icon="🛑")
            st.stop()

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

        current_scan_row, step_idx = render_scanner_chart(solver, p['a'], p['b'], p['step'], roots_list, scan_log)
        
        st.markdown("---")
        
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
                    "Interval Start": "{:.4f}", "Interval End": "{:.4f}",
                    "f(Start)": "{:.4f}", "f(End)": "{:.4f}"
                }),
                use_container_width=True, height=400,
                column_config={"Sign Change?": st.column_config.TextColumn("Sign Change?")}
            )

        st.markdown("---")
        if current_scan_row is not None:
            display_math_analysis(p['method'], current_scan_row, solver, extra_params={'step': p['step']})

    elif history:
        st.subheader("Iteration Playback")
        col_viz, col_log = st.columns([1.5, 1])
        
        with col_viz:
            row_data, step_idx = render_iteration_chart(solver, history, p['method'], p.get('g_str'))
        with col_log:
            render_iteration_table(history, step_idx, tol=p.get('tol', 0.001))

        st.markdown("---")
        display_math_analysis(p['method'], row_data, solver)