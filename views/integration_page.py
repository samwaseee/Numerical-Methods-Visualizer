import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
from algorithms_integration import IntegrationSolver
from components.int_results import render_integration_results, render_general_quadrature_formula
from components.int_graph import render_integration_chart

def show_integration_page():
    # --- NAVIGATION & CLEANUP ---
    if st.button("Back to Home", icon=":material/home:"):
        keys_to_clear = ["int_data", "int_results"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page = "landing"
        st.query_params["page"] = "landing"
        st.rerun()

    st.title("Numerical Integration")
    st.markdown("Calculate the definite integral $\int_a^b f(x) dx$ using numerical methods.")

    # --- LAYOUT ---
    col_data, col_main = st.columns([1, 2.5], gap="large")

    # --- 1. LEFT COLUMN: DATA ENTRY ---
    with col_data:
        st.subheader("Data Points")
        if "int_data" not in st.session_state:
            st.session_state.int_data = pd.DataFrame({
                "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [0.0, 0.5, 2.0, 1.5, 2.5, 1.0, 0.0]
            })

        # --- INPUT MODE TOGGLE ---
        input_method = st.radio("Choose Input Method", ["Manual Data Entry", "Generate from Function"], horizontal=True)
        
        st.dataframe(st.session_state.int_data, use_container_width=True, hide_index=True)
        
        if input_method == "Generate from Function":
            func_str = st.text_input("Function f(x)", "sin(x)", help="Use Python syntax e.g. x**2, np.sin(x)")
            c_gen1, c_gen2 = st.columns(2)
            with c_gen1: a = st.number_input("Start (a)", value=0.0, step=0.1)
            with c_gen2: b = st.number_input("End (b)", value=3.14159, step=0.1)
            n_intervals = st.number_input("Number of Intervals (N)", value=6, min_value=1, step=1)
            
            if st.button("Generate Data", use_container_width=True, icon=":material/refresh:"):
                try:
                    x_sym = sp.symbols('x')
                    # Safe parsing
                    expr = sp.sympify(func_str.replace("^", "**"))
                    f = sp.lambdify(x_sym, expr, 'numpy')
                    
                    x_vals = np.linspace(a, b, n_intervals + 1)
                    y_vals = f(x_vals)
                    
                    st.session_state.int_data = pd.DataFrame({"x": x_vals, "y": y_vals})
                    
                    # Clear results
                    if "int_results" in st.session_state: del st.session_state.int_results
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating data: {e}")

        else: # Manual Entry
            c_add = st.columns(2)
            with c_add[0]: new_x = st.number_input("x", value=0.0, step=1.0, key="int_new_x")
            with c_add[1]: new_y = st.number_input("y", value=0.0, step=1.0, key="int_new_y")
            
            if st.button("Add Point", use_container_width=True):
                new_row = pd.DataFrame([{'x': new_x, 'y': new_y}])
                st.session_state.int_data = pd.concat([st.session_state.int_data, new_row], ignore_index=True)
                if "int_results" in st.session_state: del st.session_state.int_results
                st.rerun()

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Remove Last", use_container_width=True):
                    st.session_state.int_data = st.session_state.int_data.iloc[:-1]
                    if "int_results" in st.session_state: del st.session_state.int_results
                    st.rerun()
            with b2:
                if st.button("Clear All", use_container_width=True):
                    st.session_state.int_data = pd.DataFrame(columns=['x', 'y']).astype(float)
                    if "int_results" in st.session_state: del st.session_state.int_results
                    st.rerun()

    # --- 2. RIGHT COLUMN: CONFIG & RESULTS ---
    with col_main:
        
        st.divider()
        st.subheader("Configuration")
        
        if "int_data" in st.session_state and len(st.session_state.int_data) > 1:
            N = len(st.session_state.int_data) - 1
            if N % 6 == 0:
                st.info(f"Suggestion: Since you have {N} intervals, **Weddle's Rule** will give the highest accuracy.", icon=":material/lightbulb:")
            elif N % 4 == 0:
                st.info(f"Suggestion: Since you have {N} intervals, **Boole's Rule** is highly accurate.", icon=":material/lightbulb:")
            elif N % 3 == 0:
                st.info(f"Suggestion: Since you have {N} intervals, **Simpson's 3/8 Rule** is recommended.", icon=":material/lightbulb:")
            elif N % 2 == 0:
                st.info(f"Suggestion: Since you have {N} intervals, **Simpson's 1/3 Rule** is recommended.", icon=":material/lightbulb:")
            else:
                st.info(f"Suggestion: Since you have {N} intervals, **Trapezoidal Rule** is the universal fallback.", icon=":material/lightbulb:")

        c_method, c_btn = st.columns([3, 1.5])
        
        with c_method:
            method = st.selectbox("Integration Method", [
                "Trapezoidal Rule",
                "Simpson's 1/3 Rule",
                "Simpson's 3/8 Rule",
                "Boole's Rule",
                "Weddle's Rule"
            ])
        
        with c_btn:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            calc_clicked = st.button("Calculate Integral", type="primary", icon=":material/calculate:", use_container_width=True)
        
        # Determine which derivation to show
        shown_method = None
        if calc_clicked:
            shown_method = method
        elif "int_results" in st.session_state:
            shown_method = st.session_state.int_results["method"]
            
        render_general_quadrature_formula(shown_method)
        
        if calc_clicked:
            df = st.session_state.int_data
            if df is None or len(df) < 2:
                st.error("Need at least 2 points.")
            else:
                solver = IntegrationSolver(df['x'].values, df['y'].values)
                
                res = None
                details = None
                err = None
                
                if method == "Trapezoidal Rule":
                    res, details = solver.trapezoidal_rule()
                    if res is None: err = details
                elif method == "Simpson's 1/3 Rule":
                    res, details = solver.simpson_13_rule()
                    if res is None: err = details
                elif method == "Simpson's 3/8 Rule":
                    res, details = solver.simpson_38_rule()
                    if res is None: err = details
                elif method == "Boole's Rule":
                    res, details = solver.booles_rule()
                    if res is None: err = details
                elif method == "Weddle's Rule":
                    res, details = solver.weddles_rule()
                    if res is None: err = details
                
                if err:
                    st.error(err)
                    if "int_results" in st.session_state: del st.session_state.int_results
                else:
                    st.session_state.int_results = {
                        "res": res,
                        "details": details,
                        "method": method,
                        "df": df
                    }

        # --- DISPLAY RESULTS ---
        if "int_results" in st.session_state:
            res_data = st.session_state.int_results
            
            render_integration_results(res_data['res'], res_data['details'], res_data['method'])
            render_integration_chart(res_data['df']['x'], res_data['df']['y'], res_data['method'])
