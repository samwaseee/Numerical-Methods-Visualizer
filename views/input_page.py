import streamlit as st
import sympy as sp
import re
import numpy as np
from algorithms import EquationSolver
from components.header import render_function_header

# --- URL SYNCING CALLBACKS ---
def sync_to_url(key):
    val = st.session_state[key]
    st.query_params[key] = str(val)
    st.session_state.bench_data = None

def update_x0():
    """Specific handler for x0 to sync memory and URL"""
    st.session_state.x0_val = st.session_state.input_x0
    st.query_params["x0"] = str(st.session_state.x0_val)
    st.session_state.bench_data = None

def update_bounds(new_a, new_b):
    st.session_state.range_a = float(new_a)
    st.session_state.range_b = float(new_b)
    st.query_params["a"] = str(new_a)
    st.query_params["b"] = str(new_b)

def clear_bench_data():
    st.session_state.bench_data = None

def on_function_change():
    new_func = st.session_state.func_str
    new_g = f"x - ({new_func})/1000"
    st.session_state.g_str = new_g
    
    # RESET State
    st.session_state.bench_data = None
    st.session_state.solver = None
    st.session_state.show_bounds = False 
    
    # Sync URL
    st.query_params["func"] = new_func
    st.query_params["g"] = new_g

def show_input_page(header_container):
    # --- FIX: AUTO-REPAIR BLANK G(X) ---
    # This runs every time the page loads. 
    # If g(x) is missing/empty, we regenerate it immediately.
    current_g = st.session_state.get('g_str', '')
    if not current_g or current_g.strip() == "":
        func = st.session_state.get('func_str', "x^2 - 4")
        st.session_state.g_str = f"x - ({func})/1000"
        st.query_params["g"] = st.session_state.g_str
    # -----------------------------------

    # Ensure toggle state exists
    if "show_bounds" not in st.session_state:
        st.session_state.show_bounds = False

    col_input, col_config = st.columns([1.3, 1])

    # --- LEFT COLUMN: PROBLEM DEFINITION ---
    with col_input:
        st.subheader("1. Define Problem")
        
        st.text_input(
            "Function f(x)", 
            key="func_str", 
            on_change=on_function_change
        )
        
        solver = EquationSolver(st.session_state.func_str)
        if solver.valid: 
            st.session_state.solver = solver 
            with header_container.container():
                render_function_header(solver)
            
            # --- BOUNDARY ASSISTANT SECTION ---
            st.divider()
            
            if st.button("🤖 Analyze Root Bounds (Fujiwara)", use_container_width=True):
                st.session_state.show_bounds = True
            
            if st.session_state.show_bounds:
                with st.container(border=True):
                    calculated_bound = None
                    try:
                        s_clean = solver.original_str.replace("^", "**").replace(" ", "")
                        x_sym = sp.symbols('x')
                        expr = sp.sympify(s_clean)
                        
                        if expr.is_polynomial(x_sym):
                            poly = sp.Poly(expr, x_sym)
                            coeffs = [float(c) for c in poly.all_coeffs()]
                            
                            if len(coeffs) >= 2:
                                an = abs(coeffs[0])
                                degree = len(coeffs) - 1
                                
                                fujiwara_vals = []
                                for i in range(1, degree + 1):
                                    val = abs(coeffs[i] / an)**(1/i)
                                    fujiwara_vals.append(val)
                                
                                fujiwara_bound = 2 * max(fujiwara_vals) if fujiwara_vals else 10.0
                                calculated_bound = round(fujiwara_bound, 2)
                                
                                st.markdown("### 📊 Boundary Analysis")
                                st.caption("Using **Fujiwara's Bound** (Tighter than Cauchy):")
                                st.latex(r"R = 2 \cdot \max_{1 \le i \le n} \left| \frac{a_{n-i}}{a_n} \right|^{1/i}")
                                
                                c_metric, c_btn = st.columns([2, 1])
                                with c_metric:
                                    st.metric("Suggested Interval", f"± {calculated_bound}")
                                with c_btn:
                                    st.write("") # Spacer
                                    def apply_bounds():
                                        update_bounds(-calculated_bound, calculated_bound)
                                    
                                    st.button(
                                        "Apply Interval", 
                                        on_click=apply_bounds, 
                                        icon="🎯", 
                                        use_container_width=True,
                                        key="btn_apply_bounds"
                                    )
                            else:
                                st.warning("Polynomial degree too low.")
                        else:
                            st.warning("Boundary analysis only works for Polynomials.")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            # ---------------------------------------

        else: 
            st.error("Invalid Syntax")

    # --- RIGHT COLUMN: CONFIGURATION ---
    with col_config:
        with st.container(border=True):
            st.subheader("2. Configuration")
            app_mode = st.radio(
                "Operation Mode", 
                ["Single Root Solver", "Multi-Root Scanner"], 
                horizontal=True, label_visibility="collapsed", key="app_mode", on_change=clear_bench_data 
            )
            st.divider()

            if app_mode == "Multi-Root Scanner":
                c1, c2 = st.columns(2)
                with c1: st.number_input("Start (a)", key="range_a", on_change=sync_to_url, args=("range_a",))
                with c2: st.number_input("End (b)", key="range_b", on_change=sync_to_url, args=("range_b",))
                
                p_step = st.number_input(
                    "Step Size", 
                    key="scanner_step", 
                    value=0.15,
                    min_value=0.0001, format="%.4f", step=0.05,
                    on_change=sync_to_url, args=("scanner_step",)
                )
                
                if st.button("Start Scan", type="primary", use_container_width=True, icon="🔍"):
                    if solver.valid:
                        st.session_state.bench_data = None
                        st.session_state.solver = solver
                        st.session_state.params = {
                            "method": "Find All Roots (Scan)", 
                            "a": st.session_state.range_a, 
                            "b": st.session_state.range_b, 
                            "step": p_step
                        }
                        st.session_state.page = "dashboard"
                        st.query_params["page"] = "dashboard"
                        st.rerun()

            else:
                c1, c2 = st.columns(2)
                with c1: 
                    st.number_input(
                        "Guess (x0)", 
                        value=st.session_state.x0_val, 
                        key="input_x0", 
                        on_change=update_x0
                    )
                with c2: pass 
                c3, c4 = st.columns(2)
                with c3: st.number_input("Bracket Low (a)", key="range_a", on_change=sync_to_url, args=("range_a",))
                with c4: st.number_input("Bracket High (b)", key="range_b", on_change=sync_to_url, args=("range_b",))
                
                # Input for g(x) - now guaranteed to have a value
                st.text_input("g(x) [Fixed Point]", key="g_str", on_change=sync_to_url, args=("g_str",))

                if st.button("Run Benchmark", type="primary", use_container_width=True, icon="🚀"):
                    if not solver.valid: st.error("Fix syntax first.")
                    else:
                        df = solver.run_benchmark(st.session_state.x0_val, st.session_state.range_a, st.session_state.range_b, st.session_state.g_str, tol=0.01)
                        st.session_state.bench_data = df
                        st.session_state.page = "result" 
                        st.query_params["page"] = "result"
                        st.rerun()