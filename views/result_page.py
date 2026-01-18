import streamlit as st
import numpy as np
import pandas as pd
from components.header import render_function_header

def trigger_home():
    """
    Resets EVERYTHING to default values and navigates home.
    No st.rerun() needed here (it happens automatically).
    """
    # 1. Reset Session State to Defaults
    defaults = {
        "func_str": "x^4 + 6*x^3 - 77*x^2 - 330*x + 400",
        "x0_val": 3.0,
        "range_a": -10.0,
        "range_b": 10.0,
        "scanner_step": 0.15,
        "bench_data": None,
        "solver": None,
        "page": "input"
    }
    
    for key, val in defaults.items():
        st.session_state[key] = val
        
    # 2. Reset Dependent Variables
    st.session_state.g_str = f"x - ({defaults['func_str']})/1000"
    
    # 3. Clear Browser URL (Force Reset)
    st.query_params.clear()

# --- SAFETY HELPER ---
def ensure_state_exists():
    qp = st.query_params
    if 'x0_val' not in st.session_state: 
        st.session_state.x0_val = float(qp.get("x0", 3.0))
    if 'range_a' not in st.session_state:
        st.session_state.range_a = float(qp.get("a", -10.0))
    if 'range_b' not in st.session_state:
        st.session_state.range_b = float(qp.get("b", 10.0))
    if 'g_str' not in st.session_state:
        st.session_state.g_str = qp.get("g", "x")

# --- UPDATE CALLBACKS ---
def update_x0():
    st.session_state.x0_val = st.session_state.res_x0_widget
    st.query_params["x0"] = str(st.session_state.x0_val)
    run_math()

def update_a():
    st.session_state.range_a = st.session_state.res_a_widget
    st.query_params["a"] = str(st.session_state.range_a)
    run_math()

def update_b():
    st.session_state.range_b = st.session_state.res_b_widget
    st.query_params["b"] = str(st.session_state.range_b)
    run_math()

def update_g():
    st.session_state.g_str = st.session_state.res_g_widget
    st.query_params["g"] = str(st.session_state.g_str)
    run_math()

def run_math():
    ensure_state_exists()
    solver = st.session_state.solver
    if solver:
        df = solver.run_benchmark(
            st.session_state.x0_val, 
            st.session_state.range_a, 
            st.session_state.range_b, 
            st.session_state.g_str, 
            tol=0.01
        )
        st.session_state.bench_data = df

def show_result_page(header_container):
    ensure_state_exists()
    solver = st.session_state.solver
    
    if st.session_state.bench_data is None:
        run_math()
    
    with header_container.container():
        render_function_header(solver)

    # --- THE BACK BUTTON ---
    # Clicking this now triggers the full reset defined above
    if st.button("Back to Home", icon="🏠", on_click=trigger_home): pass
    
    st.title("Benchmark Results & Analysis")

    if st.session_state.bench_data is not None:
        df = st.session_state.bench_data
        def style_status(val):
            return 'color: green; font-weight: bold' if val == "Converged" else 'color: red'
        st.dataframe(
            df.style.applymap(style_status, subset=['Status']).format({"Error %": "{:.4f}", "Root Found": "{:.4f}"}), 
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("🛠️ Modify Parameters")
    
    if st.session_state.bench_data is not None:
        fixed_row = st.session_state.bench_data[st.session_state.bench_data['Method'] == "Fixed Point"]
        if not fixed_row.empty and "Converged" not in fixed_row.iloc[0]['Status']:
            try:
                deriv = float(solver.df(st.session_state.x0_val))
                denom = int(deriv) if abs(deriv) > 1 else (1 if deriv >= 0 else -1)
                if denom == 0: denom = 1
                current_func = st.session_state.func_str
                st.warning(
                    f"⚠️ **Fixed Point Diverged?**\n\n"
                    f"The derivative $f'({st.session_state.x0_val})$ is **{deriv:.2f}**.\n"
                    f"For convergence, the denominator must match this sign.\n\n"
                    f"👉 **Suggested Fix:** Change $g(x)$ to:\n"
                    f"`x - ({current_func}) / {denom}`",
                    icon="💡"
                )
            except: pass

    col_params, col_viz = st.columns([1, 1.5])

    with col_params:
        with st.container(border=True):
            st.markdown("**Adjust Inputs**")
            
            st.number_input(
                "Guess (x0)", 
                value=st.session_state.x0_val, 
                key="res_x0_widget", 
                on_change=update_x0
            )
            
            c1, c2 = st.columns(2)
            with c1: 
                st.number_input(
                    "Lower (a)", 
                    value=st.session_state.range_a, 
                    key="res_a_widget", 
                    on_change=update_a
                )
            with c2: 
                st.number_input(
                    "Upper (b)", 
                    value=st.session_state.range_b, 
                    key="res_b_widget", 
                    on_change=update_b
                )
            
            st.text_input(
                "g(x)", 
                value=st.session_state.g_str, 
                key="res_g_widget", 
                on_change=update_g
            )
            
            if st.button("Recalculate Table", icon="🔄", use_container_width=True):
                run_math()
                st.rerun()

    with col_viz:
        st.markdown("**Select Method to Visualize**")
        vis_iters = st.number_input("Max Iterations for Graph", value=100)
        
        methods = ["Newton-Raphson", "Bisection Method", "False Position", "Secant Method", "Fixed Point Iteration"]
        
        cols = st.columns(3)
        for i, m in enumerate(methods):
            if cols[i % 3].button(m, use_container_width=True):
                base_params = {"method": m, "iters": vis_iters, "tol": 0.01}
                st.query_params["page"] = "dashboard" 
                
                if m in ["Newton-Raphson", "Secant Method"]:
                    base_params["x0"] = st.session_state.x0_val
                    if m == "Secant Method": base_params["x1"] = st.session_state.x0_val + 1.0
                elif m in ["Bisection Method", "False Position"]:
                    base_params["a"] = st.session_state.range_a
                    base_params["b"] = st.session_state.range_b
                elif m == "Fixed Point Iteration":
                    base_params["x0"] = st.session_state.x0_val
                    base_params["g_str"] = st.session_state.g_str
                
                st.session_state.params = base_params
                st.session_state.page = "dashboard"
                st.rerun()

        st.markdown("---")
        
        # --- NEW: COMPARISON MODE TRIGGER ---
        st.subheader("⚔️ Compare Two Methods")
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            m1 = st.selectbox("Method A", methods, index=0)
        with c_m2:
            m2 = st.selectbox("Method B", methods, index=1)
            
        if st.button("Start Comparison Race", type="primary", use_container_width=True):
            st.session_state.params = {
                "method": "Comparison Mode",
                "method_a": m1,
                "method_b": m2,
                "iters": vis_iters,
                "x0": st.session_state.x0_val,
                "a": st.session_state.range_a,
                "b": st.session_state.range_b,
                "g_str": st.session_state.g_str
            }
            st.session_state.page = "dashboard"
            st.query_params["page"] = "dashboard"
            st.rerun()