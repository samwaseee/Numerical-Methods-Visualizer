import streamlit as st
import pandas as pd
import numpy as np
from algorithms_differentiation import DifferentiationSolver
from algorithms_interpolation import InterpolationSolver
from components.diff_results import render_differentiation_results
from components.diff_graph import render_differentiation_chart

def show_differentiation_page():
    # --- NAVIGATION & CLEANUP ---
    if st.button("Back to Home", icon=":material/home:"):
        keys_to_clear = ["diff_data", "diff_results", "true_func_str"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page = "landing"
        st.query_params["page"] = "landing"
        st.rerun()

    st.title("Numerical Differentiation")
    st.markdown("Calculate the slope ($1^{st}$ Derivative) and curvature ($2^{nd}$ Derivative) of data points.")

    # --- LAYOUT ---
    col_data, col_main = st.columns([1, 2.5], gap="large")

    # --- 1. LEFT COLUMN: DATA ENTRY ---
    with col_data:
        st.subheader("Data Points")
        if "diff_data" not in st.session_state:
            st.session_state.diff_data = pd.DataFrame(
                {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [1.0, 2.718, 7.389, 20.08, 54.59, 148.4, 403.4]}
            )

        # --- INPUT MODE TOGGLE ---
        input_mode = st.radio("Input Mode", ["Manual Entry", "Generate from Function"], horizontal=True)

        st.dataframe(st.session_state.diff_data, use_container_width=True, hide_index=True)

        if input_mode == "Generate from Function":
            func_str = st.text_input("Function f(x)", value="sin(x)", help="Use numpy syntax: sin(x), exp(x), x**2")
            c_gen1, c_gen2 = st.columns(2)
            with c_gen1: start_x = st.number_input("Start X", value=0.0, step=0.1)
            with c_gen2: end_x = st.number_input("End X", value=6.0, step=0.1)
            n_points = st.slider("Number of Points", 3, 100, 7)
            
            if st.button("Generate Data", use_container_width=True, icon=":material/refresh:"):
                try:
                    # Safe evaluation environment
                    safe_dict = {k: getattr(np, k) for k in dir(np) if not k.startswith('_')}
                    x_vals = np.linspace(start_x, end_x, n_points)
                    safe_dict['x'] = x_vals
                    
                    # Parse and evaluate
                    expr = func_str.replace("^", "**")
                    y_vals = eval(expr, {"__builtins__": {}}, safe_dict)
                    
                    # Handle constant functions
                    if np.isscalar(y_vals):
                        y_vals = np.full_like(x_vals, y_vals)
                        
                    st.session_state.diff_data = pd.DataFrame({"x": x_vals, "y": y_vals})
                    st.session_state.true_func_str = func_str
                    
                    # Clear results
                    if "diff_results" in st.session_state: del st.session_state.diff_results
                    st.rerun()
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        if input_mode == "Manual Entry":
            c_add = st.columns(2)
            with c_add[0]: new_x = st.number_input("x", value=0.0, step=1.0, key="diff_new_x")
            with c_add[1]: new_y = st.number_input("y", value=0.0, step=1.0, key="diff_new_y")
            
            if st.button("Add Point", use_container_width=True):
                new_row = pd.DataFrame([{'x': new_x, 'y': new_y}])
                st.session_state.diff_data = pd.concat([st.session_state.diff_data, new_row], ignore_index=True)
                if "diff_results" in st.session_state: del st.session_state.diff_results
                if "true_func_str" in st.session_state: del st.session_state.true_func_str
                st.rerun()

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Remove Last", use_container_width=True):
                    st.session_state.diff_data = st.session_state.diff_data.iloc[:-1]
                    if "diff_results" in st.session_state: del st.session_state.diff_results
                    if "true_func_str" in st.session_state: del st.session_state.true_func_str
                    st.rerun()
            with b2:
                if st.button("Clear All", use_container_width=True):
                    st.session_state.diff_data = pd.DataFrame(columns=['x', 'y']).astype(float)
                    if "diff_results" in st.session_state: del st.session_state.diff_results
                    if "true_func_str" in st.session_state: del st.session_state.true_func_str
                    st.rerun()

    # --- 2. RIGHT COLUMN: CONFIG & RESULTS ---
    with col_main:
        category = st.radio(
            "Differentiation Category",
            ["Equal Intervals", "Unequal Intervals"],
            horizontal=True
        )
        
        st.divider()
        
        c_method, c_target, c_btn = st.columns([2, 1, 1])
        
        with c_method:
            if category == "Equal Intervals":
                method = st.selectbox("Method", ["Newton Forward", "Newton Backward", "Stirling's Formula", "Bessel's Formula", "Richardson Extrapolation"])
            else:
                method = st.selectbox("Method", ["Lagrange Polynomial", "Newton Divided Difference", "Cubic Spline", "Least Squares (Noisy Data)"])
        
        with c_target:
            target_x = st.number_input("Target x", value=3.0, step=0.1, format="%.4f")
            
        with c_btn:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            calc_clicked = st.button("Calculate", type="primary", icon=":material/calculate:", use_container_width=True)
        
        if calc_clicked:
            df_data = st.session_state.diff_data.dropna().astype(float)
            if len(df_data) < 2:
                st.error("Need at least 2 points.")
            else:
                solver = DifferentiationSolver(df_data['x'].values, df_data['y'].values)
                res = None
                df_table = None
                err = None
                
                if method == "Newton Forward":
                    res, df_table, err = solver.newton_forward_diff(target_x)
                elif method == "Newton Backward":
                    res, df_table, err = solver.newton_backward_diff(target_x)
                elif method == "Stirling's Formula":
                    res, df_table, err = solver.stirling_diff(target_x)
                elif method == "Bessel's Formula":
                    res, df_table, err = solver.bessel_diff(target_x)
                elif method == "Richardson Extrapolation":
                    res, df_table, err = solver.richardson_diff(target_x)
                elif method == "Cubic Spline":
                    res, df_table, err = solver.cubic_spline_diff(target_x)
                elif method == "Least Squares (Noisy Data)":
                    res, df_table, err = solver.least_squares_diff(target_x)
                elif method == "Lagrange Polynomial":
                    res, df_table, err = solver.lagrange_derivative(target_x)
                elif method == "Newton Divided Difference":
                    res, df_table, err = solver.newton_divided_difference_diff(target_x)
                
                if err:
                    st.error(err)
                    if "diff_results" in st.session_state:
                        del st.session_state.diff_results
                else:
                    # Find Extrema (Minima/Maxima)
                    extrema = []
                    try:
                        extrema = solver.find_extrema()
                    except Exception:
                        pass

                    st.session_state.diff_results = {
                        "res": res,
                        "table": df_table,
                        "method": method,
                        "df_data": df_data,  # Save original data for 'h' calc
                        "extrema": extrema
                    }
                    
        # --- DISPLAY RESULTS ---
        if "diff_results" in st.session_state:
            results_data = st.session_state.diff_results
            if "res" not in results_data or results_data['res'] is None:
                if "diff_results" in st.session_state:
                    del st.session_state.diff_results
                st.rerun()

            res = results_data['res']
            df_table = results_data['table']
            method = results_data['method']
            orig_data = results_data.get('df_data', pd.DataFrame())
            extrema = results_data.get('extrema', [])

            render_differentiation_results(res, df_table, method, orig_data, target_x, extrema)
            render_differentiation_chart(orig_data, target_x, res, method, extrema)