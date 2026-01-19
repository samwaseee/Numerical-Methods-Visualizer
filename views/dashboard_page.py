import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from components.header import render_function_header
from components.math_display import display_math_analysis
from components.graph_plotter import render_scanner_chart, render_iteration_chart, render_complex_plane_chart
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
    # Helper to extract 'x' from different dict keys for different methods
    def get_x(row): return row.get('x_new', row.get('c (root)', row.get('p_real', 0)))

    # Determine bounds
    all_x = []
    if hist_a: all_x.extend([get_x(r) for r in hist_a])
    if hist_b: all_x.extend([get_x(r) for r in hist_b])
    
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
        path_x = [get_x(r) for r in hist_a]
        path_y = [solver.f(x) for x in path_x]
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y, 
            mode='lines+markers', name=name_a,
            line=dict(color="#FB002A", dash='dot'),
            marker=dict(symbol='circle', size=8)
        ))

    # Plot Method B (Squares)
    if hist_b:
        path_x = [get_x(r) for r in hist_b]
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
    
    # --- HELPER TO RUN METHODS ---
    def run_method(name, params):
        try:
            history = []
            if name == "Newton-Raphson": 
                history = solver.newton_raphson_method(params['x0'], params['iters'])
            elif name == "Bisection Method": 
                history = solver.bisection_method(params['a'], params['b'], params['iters'])
            elif name == "False Position": 
                history = solver.false_position_method(params['a'], params['b'], params['iters'])
            elif name == "Secant Method": 
                history = solver.secant_method(params['x0'], params['x0']+1, params['iters'])
            elif name == "Fixed Point Iteration": 
                res = solver.fixed_point_method(params['g_str'], params['x0'], params['iters'])
                history = res[0] if res else []
            # --- NEW METHODS ---
            elif name == "Brent's Method": 
                history = solver.brents_method(params['a'], params['b'], params['iters'])
            elif name == "Halley's Method": 
                history = solver.halleys_method(params['x0'], params['iters'])
            elif name == "Muller's Method": 
                history = solver.mullers_method(params['x0'], params['x0']+0.5, params['x0']+1, params['iters'])
            
            # Normalize history for consistent column format
            return solver.normalize_history(history, name)
        except: return []

    # --- COMPARISON MODE LOGIC ---
    if p.get('method') == "Comparison Mode":
        st.title("⚔️ Method Comparison Race")
        
        hist_a = run_method(p['method_a'], p)
        hist_b = run_method(p['method_b'], p)

        render_comparison_chart(solver, hist_a, p['method_a'], hist_b, p['method_b'])

        # 4. Render Stats Table
        st.subheader("📊 Performance Analysis")
        col1, col2 = st.columns(2)
        
        def show_stat(col, name, hist):
            with col:
                st.markdown(f"**{name}**")
                if hist:
                    last = hist[-1]
                    root = last.get('x_new', last.get('c (root)', last.get('p_real', 0)))
                    err = last.get('Error % (Ea)', 0)
                    st.metric("Iterations", len(hist))
                    st.metric("Final Error", f"{err:.4f}%")
                    if isinstance(root, complex):
                        real_part = round(float(root.real), 4)
                        imag_part = round(float(root.imag), 4)
                        st.metric("Final Root", f"{real_part:.4f} + {imag_part:.4f}i")
                    else:
                        st.metric("Final Root", f"{root:.4f}")
                else: st.error("Failed")

        show_stat(col1, p['method_a'], hist_a)
        show_stat(col2, p['method_b'], hist_b)
        return
    # -----------------------------

    st.title(f"{p['method']}")

    history = []
    roots_list = []
    scan_log = []
    
    if p['method'] == "Find All Roots (Scan)":
        roots_list, scan_log = solver.find_all_roots(p['a'], p['b'], p['step'])
    else:
        history = run_method(p['method'], p)

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

        # --- DISPLAY ROOT FOUND ---
        st.markdown("---")
        root_found = row_data.get('x_new', row_data.get('c (root)', None))
        
        # Check if root is complex (for Muller's method)
        is_complex_root = isinstance(root_found, complex) or (hasattr(np, 'iscomplex') and np.iscomplex(root_found))
        
        if root_found is not None:
            col_root, col_verify = st.columns([1, 1])
            with col_root:
                if is_complex_root:
                    st.markdown(f"<h3 style='text-align: center; color: #f43f5e;'>✓ Complex Root Found</h3>", unsafe_allow_html=True)
                    real_part = root_found.real if hasattr(root_found, 'real') else float(root_found)
                    imag_part = root_found.imag if hasattr(root_found, 'imag') else 0
                    st.markdown(f"<h1 style='text-align: center; color: #f43f5e; font-weight: bold;'>{real_part} + {imag_part}i</h1>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='text-align: center; color: #10b981;'>✓ Root Found</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: center; color: #10b981; font-weight: bold;'>{root_found:.4f}</h1>", unsafe_allow_html=True)
            with col_verify:
                try:
                    f_val = solver.f(root_found)
                    st.markdown(f"<h3 style='text-align: center;'>Verification</h3>", unsafe_allow_html=True)
                    if is_complex_root:
                        real_part = root_found.real if hasattr(root_found, 'real') else float(root_found)
                        imag_part = root_found.imag if hasattr(root_found, 'imag') else 0
                        f_real = f_val.real if hasattr(f_val, 'real') else float(f_val)
                        f_imag = f_val.imag if hasattr(f_val, 'imag') else 0
                        st.markdown(f"<p style='text-align: center; font-size: 18px;'>f({real_part} + {imag_part}i)</p>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='text-align: center; color: #3b82f6;'>{f_real} + {f_imag}i</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='text-align: center; font-size: 18px;'>f({root_found:.4f})</p>", unsafe_allow_html=True)
                        st.markdown(f"<h2 style='text-align: center; color: #3b82f6;'>{f_val:.4e}</h2>", unsafe_allow_html=True)
                except:
                    pass
        
        # --- DISPLAY COMPLEX ROOTS FOR MULLER'S METHOD ---
        if p['method'] == "Muller's Method" and history:
            st.markdown("---")
            st.subheader("Complex Roots Visualization")
            
            # Extract roots only up to the selected step (respecting slider)
            history_df = pd.DataFrame(history)
            history_up_to_step = history_df.iloc[:step_idx]
            
            all_roots = []
            for _, row in history_up_to_step.iterrows():
                root = row.get('x_new')
                if root is not None:
                    all_roots.append(root)
            
            # Filter complex roots
            complex_roots = [r for r in all_roots if isinstance(r, complex) or np.iscomplex(r)]
            
            if complex_roots:
                render_complex_plane_chart(complex_roots)
            else:
                st.info("No complex roots found for this function.")
        
        st.markdown("---")
        display_math_analysis(p['method'], row_data, solver)