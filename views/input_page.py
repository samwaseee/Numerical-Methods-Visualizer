import streamlit as st
import sympy as sp
import re
import numpy as np
from algorithms import EquationSolver
from components.header import render_function_header

def vspace(height_px):
    """Creates a vertical gap of specific height in pixels."""
    st.markdown(f'<div style="height: {height_px}px;"></div>', unsafe_allow_html=True)

# --- URL SYNCING CALLBACKS ---
def sync_to_url(key):
    val = st.session_state[key]
    st.query_params[key] = str(val)
    st.session_state.bench_data = None

def update_x0(new_val=None):
    """Specific handler for x0 to sync memory and URL"""
    if new_val is not None:
        val = float(new_val)
        st.session_state.x0_val = val
        # --- CRITICAL FIX: Update the widget key directly ---
        st.session_state.input_x0 = val 
        # --------------------------------------------------
    else:
        # If triggered by the widget itself, read from the key
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
            
            # --- POLYNOMIAL ANALYSIS ASSISTANT ---
            st.divider()
            
            if st.button("🤖 Polynomial Analysis (Bounds & Guess)", use_container_width=True):
                st.session_state.show_bounds = True
            
            if st.session_state.show_bounds:
                with st.container(border=True):
                    st.markdown("### 🧬 Polynomial Analysis")
                    st.caption("These tools work **only for Polynomials** (e.g., $ax^n + \dots$).")
                    
                    try:
                        s_clean = solver.original_str.replace("^", "**").replace(" ", "")
                        x_sym = sp.symbols('x')
                        expr = sp.sympify(s_clean)
                        
                        if expr.is_polynomial(x_sym):
                            poly = sp.Poly(expr, x_sym)
                            coeffs = [float(c) for c in poly.all_coeffs()]
                            degree = len(coeffs) - 1

                            if len(coeffs) >= 2:
                                # --- MATH CALCULATIONS ---
                                
                                # 1. FUJIWARA BOUND (Root isolation radius)
                                an = abs(coeffs[0])
                                fujiwara_vals = []
                                for i in range(1, degree + 1):
                                    val = abs(coeffs[i] / an)**(1/i)
                                    fujiwara_vals.append(val)
                                fujiwara_bound = 2 * max(fujiwara_vals) if fujiwara_vals else 10.0
                                calculated_bound = round(fujiwara_bound, 2)
                                
                                # 2. CENTROID OF ROOTS (For Initial Guess)
                                an_1 = coeffs[1] if len(coeffs) > 1 else 0
                                suggested_x0 = -an_1 / (degree * coeffs[0])
                                suggested_x0 = round(suggested_x0, 2)
                                
                                # 3. DYNAMIC BRACKET SEARCH (Find interval with sign change)
                                def find_best_bracket(f, center, radius, steps=20):
                                    """Search for interval [a,b] where f(a)*f(b) < 0"""
                                    best_bracket = None
                                    best_span = float('inf')
                                    
                                    test_points = np.linspace(center - radius, center + radius, steps)
                                    f_vals = []
                                    for pt in test_points:
                                        try:
                                            f_vals.append((pt, float(f(pt))))
                                        except:
                                            f_vals.append((pt, np.inf))
                                    
                                    # Search for sign changes
                                    for i in range(len(f_vals) - 1):
                                        x1, y1 = f_vals[i]
                                        x2, y2 = f_vals[i + 1]
                                        
                                        if y1 * y2 < 0:  # Sign change found!
                                            span = x2 - x1
                                            if span < best_span:
                                                best_span = span
                                                best_bracket = (round(x1, 2), round(x2, 2))
                                    
                                    return best_bracket

                                # Find best bracket around the Fujiwara bound
                                best_bracket = find_best_bracket(solver.f, suggested_x0, fujiwara_bound)

                                # --- DISPLAY UI (COLUMN LAYOUT) ---
                                col_bracket, col_guess = st.columns(2)

                                with col_bracket:
                                    st.markdown("#### 🎯 Bracketing")
                                    st.caption("**Fujiwara's Bound:** (root isolation radius)")
                                    st.latex(r"R = 2 \cdot \max \left| \frac{a_{n-i}}{a_n} \right|^{1/i}")
                                    
                                    if best_bracket:
                                        a, b = best_bracket
                                        st.metric("Suggested Bracket", f"[{a}, {b}]")
                                        if st.button("Apply Bounds", use_container_width=True):
                                            update_bounds(a, b)
                                            st.session_state.bench_data = None
                                    else:
                                        st.warning(f"⚠️ No sign change found in [{-calculated_bound}, {calculated_bound}]")

                                with col_guess:
                                    st.markdown("#### 💡 Initial Guess")
                                    st.caption("**Centroid of Roots:**")
                                    st.latex(r"x_0 = -\frac{a_{n-1}}{n \cdot a_n}")
                                    vspace(0)
                                    st.metric("Suggested Start", f"{suggested_x0}")
                                    if st.button("Apply Guess", use_container_width=True):
                                        update_x0(suggested_x0)
                                        st.rerun()
                                        
                            else:
                                st.warning("Polynomial degree too low.")
                        else:
                            st.warning("⚠️ Analysis skipped. Function is not a polynomial.")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            # ---------------------------------------

        else: 
            st.error("Invalid Syntax")

    # --- RIGHT COLUMN: CONFIGURATION ---
    with col_config:
        with st.container(border=True):
            st.subheader("2. Configuration")
            st.markdown(
                """
        <style>
        /* 1. Target the flex container of the radio button option */
        div[role="radiogroup"] label {
            display: flex !important;
            align-items: center !important;
        }

        /* 2. Target the text inside the radio button */
        div[role="radiogroup"] label p {
            font-size: 24px !important;
            font-weight: 600 !important;
            margin-bottom: 0px !important;
        }
        
        </style>
            """,
            unsafe_allow_html=True)
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
                        on_change=lambda: update_x0(None)
                    )
                with c2: pass 
                c3, c4 = st.columns(2)
                with c3: st.number_input("Bracket Low (a)", key="range_a", on_change=sync_to_url, args=("range_a",))
                with c4: st.number_input("Bracket High (b)", key="range_b", on_change=sync_to_url, args=("range_b",))
                
                st.text_input("g(x) [Fixed Point]", key="g_str", on_change=sync_to_url, args=("g_str",))

                if st.button("Run Benchmark", type="primary", use_container_width=True, icon="🚀"):
                    if not solver.valid: st.error("Fix syntax first.")
                    else:
                        df = solver.run_benchmark(st.session_state.x0_val, st.session_state.range_a, st.session_state.range_b, st.session_state.g_str, tol=0.01)
                        st.session_state.bench_data = df
                        st.session_state.page = "result" 
                        st.query_params["page"] = "result"
                        st.rerun()