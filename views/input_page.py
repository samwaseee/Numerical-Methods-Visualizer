import streamlit as st
import sympy as sp
import re
import numpy as np
from algorithms import EquationSolver
from components.header import render_function_header

# --- STORAGE & STATE INITIALIZATION ---
def get_from_url(key, default, cast_func=str):
    """Safely gets a value from the URL. Treats empty strings as Default."""
    qp = st.query_params
    if key in qp:
        try:
            val = cast_func(qp[key])
            return val if val else default 
        except:
            return default
    return default

def initialize_session_state():
    # Page State
    if 'page' not in st.session_state:
        st.session_state.page = get_from_url("page", "landing")

    # Inputs (with robust defaults)
    if 'func_str' not in st.session_state: 
        st.session_state.func_str = get_from_url("func", "x^4 + 6*x^3 - 77*x^2 - 330*x + 400")

    if 'x0_val' not in st.session_state: 
        st.session_state.x0_val = get_from_url("x0", 3.0, float)

    if 'range_a' not in st.session_state: 
        st.session_state.range_a = get_from_url("a", -10.0, float)

    if 'range_b' not in st.session_state: 
        st.session_state.range_b = get_from_url("b", 10.0, float)

    if 'g_str' not in st.session_state: 
        url_g = get_from_url("g", "")
        st.session_state.g_str = url_g if url_g else f"x - ({st.session_state.func_str})/1000"

    if 'scanner_step' not in st.session_state:
        st.session_state.scanner_step = get_from_url("step", 0.15, float)

    if 'solver' not in st.session_state: st.session_state.solver = None
    if 'bench_data' not in st.session_state: st.session_state.bench_data = None

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

def update_from_widget(storage_key, widget_key):
    st.session_state[storage_key] = st.session_state[widget_key]
    st.query_params[storage_key] = str(st.session_state[storage_key])
    st.session_state.bench_data = None

def update_bounds(new_a, new_b):
    st.session_state.range_a = float(new_a)
    st.session_state.range_b = float(new_b)
    st.session_state.input_a = float(new_a)
    st.session_state.input_b = float(new_b)
    st.query_params["a"] = str(new_a)
    st.query_params["b"] = str(new_b)

def clear_bench_data():
    st.session_state.bench_data = None

def on_function_change():
    if "input_func" in st.session_state:
        st.session_state.func_str = st.session_state.input_func

    new_func = st.session_state.func_str
    new_g = f"x - ({new_func})/1000"
    st.session_state.g_str = new_g
    st.session_state.input_g = new_g
    
    # RESET State
    st.session_state.bench_data = None
    st.session_state.solver = None
    st.session_state.show_bounds = False 
    
    # Sync URL
    st.query_params["func"] = new_func
    st.query_params["g"] = new_g

def show_input_page(header_container):
    initialize_session_state()

    if st.button("Back to Home", icon=":material/home:"):
        st.session_state.page = "landing"
        st.query_params["page"] = "landing"
        st.rerun()

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

    st.subheader("Define Problem")
    
    st.text_input(
        "Function f(x)", 
        value=st.session_state.func_str,
        key="input_func", 
        on_change=on_function_change
    )
    
    solver = EquationSolver(st.session_state.func_str)
    if solver.valid: 
        st.session_state.solver = solver 
        with header_container.container():
            render_function_header(solver)
    else:
        st.error("Invalid Syntax")

    st.divider()

    col_analysis, col_config = st.columns([1.3, 1])

    with col_analysis:
        if solver.valid:
            if st.button("Polynomial Analysis (Bounds & Guess)", use_container_width=True, icon=":material/analytics:"):
                st.session_state.show_bounds = True
            
            if st.session_state.show_bounds:
                with st.container(border=True):
                    st.markdown("### Polynomial Analysis")
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
                                    st.markdown("#### Bracketing")
                                    st.caption("**Fujiwara's Bound:** (root isolation radius)")
                                    st.latex(r"R = 2 \cdot \max \left| \frac{a_{n-i}}{a_n} \right|^{1/i}")
                                    
                                    st.metric("Theoretical Range (All Roots)", f"[{-calculated_bound}, {calculated_bound}]")
                                    if st.button("Apply Theoretical Range", use_container_width=True, key="btn_theo"):
                                        update_bounds(-calculated_bound, calculated_bound)
                                        st.session_state.bench_data = None
                                    
                                    # st.divider()

                                    if best_bracket:
                                        a, b = best_bracket
                                        st.metric("Sign Change Bracket", f"[{a}, {b}]")
                                        if st.button("Apply Sign Change Bracket", use_container_width=True, key="btn_sign"):
                                            update_bounds(a, b)
                                            st.session_state.bench_data = None
                                    else:
                                        st.warning(f"No sign change found in [{-calculated_bound}, {calculated_bound}]", icon=":material/warning:")

                                with col_guess:
                                    st.markdown("#### Initial Guess")
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
                            st.warning("Analysis skipped. Function is not a polynomial.", icon=":material/warning:")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            # ---------------------------------------
    
    with col_config:
        with st.container(border=True):
            st.subheader("Configuration")
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
                with c1: st.number_input("Start (a)", value=st.session_state.range_a, key="input_a", on_change=update_from_widget, args=("range_a", "input_a"))
                with c2: st.number_input("End (b)", value=st.session_state.range_b, key="input_b", on_change=update_from_widget, args=("range_b", "input_b"))
                
                p_step = st.number_input(
                    "Step Size", 
                    value=st.session_state.scanner_step,
                    key="input_step", 
                    min_value=0.0001, format="%.4f", step=0.05,
                    on_change=update_from_widget, args=("scanner_step", "input_step")
                )
                
                if st.button("Start Scan", type="primary", use_container_width=True, icon=":material/search:"):
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
                with c3: st.number_input("Bracket Low (a)", value=st.session_state.range_a, key="input_a", on_change=update_from_widget, args=("range_a", "input_a"))
                with c4: st.number_input("Bracket High (b)", value=st.session_state.range_b, key="input_b", on_change=update_from_widget, args=("range_b", "input_b"))
                
                st.text_input("g(x) [Fixed Point]", value=st.session_state.g_str, key="input_g", on_change=update_from_widget, args=("g_str", "input_g"))

                if st.button("Run Benchmark", type="primary", use_container_width=True, icon=":material/rocket_launch:"):
                    if not solver.valid: st.error("Fix syntax first.")
                    else:
                        df = solver.run_benchmark(st.session_state.x0_val, st.session_state.range_a, st.session_state.range_b, st.session_state.g_str, tol=0.01)
                        st.session_state.bench_data = df
                        st.session_state.page = "result" 
                        st.query_params["page"] = "result"
                        st.rerun()