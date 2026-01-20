import streamlit as st
import pandas as pd
from algorithms_interpolation import InterpolationSolver
from components.interp_results import render_result
from components.interp_graph import render_interpolation_chart

def show_interpolation_page():
    if st.button("Back to Home", icon=":material/home:"):
        # 1. LIST the specific keys you want to kill
        keys_to_clear = ["interp_data", "results_ready", "editor"]

        # 2. DELETE them if they exist
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # 3. NOW navigate away
        st.session_state.page = "landing"
        st.query_params["page"] = "landing"
        st.rerun()
        
        
    st.title("Interpolation Tool")
    
    if "results_ready" not in st.session_state:
        st.session_state.results_ready = False

    def reset_results():
        st.session_state.results_ready = False

    # --- LAYOUT ---
    col_data, col_main = st.columns([1, 2.5], gap="large")
    
    # --- 1. LEFT COLUMN: DATA ENTRY ---
    with col_data:
        st.subheader("Data Points")
        if "interp_data" not in st.session_state:
            st.session_state.interp_data = pd.DataFrame(
                {"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [1.0, 12.0, 5.0, 8.0, 15.0], "yp": [0.0]*5}
            )
        
        # Ensure yp column exists (migration for existing sessions)
        if "yp" not in st.session_state.interp_data.columns:
            st.session_state.interp_data["yp"] = 0.0

        # Determine if we need to show yp
        # We need to peek at the method selection, but it's defined later. 
        # We'll use a session state flag or just check the widget if it exists, defaulting to False.
        show_yp = st.session_state.get("is_hermite", False)
        
        cols_to_show = ["x", "y", "yp"] if show_yp else ["x", "y"]
        st.dataframe(st.session_state.interp_data[cols_to_show], use_container_width=True, hide_index=True)

        # Add Point
        c_add = st.columns(3 if show_yp else 2)
        with c_add[0]: new_x = st.number_input("x", value=0.0, step=1.0, key="new_x")
        with c_add[1]: new_y = st.number_input("y", value=0.0, step=1.0, key="new_y")
        new_yp = 0.0
        if show_yp:
            with c_add[2]: new_yp = st.number_input("y'", value=0.0, step=1.0, key="new_yp")
        
        if st.button("Add Point", use_container_width=True):
            new_row = pd.DataFrame([{'x': new_x, 'y': new_y, 'yp': new_yp}])
            st.session_state.interp_data = pd.concat([st.session_state.interp_data, new_row], ignore_index=True)
            reset_results()
            st.rerun()
            
        # Controls
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Remove Last", use_container_width=True):
                st.session_state.interp_data = st.session_state.interp_data.iloc[:-1]
                reset_results()
                st.rerun()
        with b2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.interp_data = pd.DataFrame(columns=['x', 'y', 'yp']).astype(float)
                reset_results()
                st.rerun()

    # --- 2. RIGHT COLUMN: CONFIG & RESULTS ---
    with col_main:
        # --- CATEGORY SELECTOR ---
        category = st.radio(
            "Interpolation Category",
            ["Equal Intervals", "Unequal Intervals", "Inverse Interpolation"],
            horizontal=True,
            on_change=reset_results
        )
        
        st.divider()
        
        # --- METHOD SELECTOR BASED ON CATEGORY ---
        method = None
        
        if category == "Equal Intervals":
            method = st.selectbox("Method", [
                "Newton Forward", 
                "Newton Backward", 
                "Gauss Forward", 
                "Gauss Backward",
                "Stirling's Formula",
                "Bessel's Formula"
            ], on_change=reset_results)
            
        elif category == "Unequal Intervals":
            method = st.selectbox("Method", [
                "Lagrange Interpolation", 
                "Newton Divided Difference",
                "Hermite Interpolation",
                "Cubic Spline Interpolation"
            ], on_change=reset_results)
            
        elif category == "Inverse Interpolation":
            st.info("Finds x for a given y. Swaps axes (Input Y -> Output X).")
            method = st.selectbox("Method", [
                "Inverse Lagrange",
                "Inverse Newton Divided Difference"
            ], on_change=reset_results)

        # Update Hermite Flag for UI
        if method == "Hermite Interpolation":
            if not st.session_state.get("is_hermite", False):
                st.session_state.is_hermite = True
                st.rerun()
        else:
            if st.session_state.get("is_hermite", False):
                st.session_state.is_hermite = False
                st.rerun()

        # --- CALCULATE BUTTON ---
        if st.button("Calculate Polynomial", type="primary", icon=":material/calculate:"):
            st.session_state.results_ready = True

        # --- EXECUTION LOGIC ---
        if st.session_state.results_ready:
            df_data = st.session_state.interp_data
            if len(df_data) < 2:
                st.error("Need at least 2 points.")
                return

            # Initialize Solver
            # Handle Inverse: Swap X and Y
            if category == "Inverse Interpolation":
                # For inverse interpolation, 'y' values must be unique
                if df_data['y'].duplicated().any():
                    st.error(
                        "Inverse Interpolation requires all 'y' values to be unique. "
                        "Please edit your data to remove duplicate y-entries.",
                        icon="❗"
                    )
                    return
                solver = InterpolationSolver(df_data['y'].values, df_data['x'].values)
                is_inverse = True
            else:
                # For standard interpolation, 'x' values must be unique
                if df_data['x'].duplicated().any():
                    st.error(
                        "Interpolation requires all 'x' values to be unique. "
                        "Please edit your data to remove duplicate x-entries.",
                        icon="❗"
                    )
                    return
                solver = InterpolationSolver(df_data['x'].values, df_data['y'].values)
                is_inverse = False

            # Run Selected Method
            df_table = None
            expr = None
            err = None
            
            if method == "Newton Forward":
                df, expr, err = solver.newton_forward()
            elif method == "Newton Backward":
                df, expr, err = solver.newton_backward()
            elif method == "Gauss Forward":
                df, expr, err = solver.gauss_forward_method()
            elif method == "Gauss Backward":
                df, expr, err = solver.gauss_backward_method()
            elif method == "Stirling's Formula":
                df, expr, err = solver.stirling_method()
            elif method == "Bessel's Formula":
                df, expr, err = solver.bessel_method()
            elif method == "Lagrange Interpolation" or method == "Inverse Lagrange":
                df, expr = solver.lagrange_method()
            elif method == "Newton Divided Difference":
                df, expr = solver.newton_divided_difference()
            elif method == "Hermite Interpolation":
                df, expr, err = solver.hermite_interpolation(df_data['yp'].values)
            elif method == "Cubic Spline Interpolation":
                df, expr, err = solver.cubic_spline_interpolation()
            elif method == "Inverse Newton Divided Difference":
                df, expr = solver.newton_divided_difference()

            # Display Results
            if err:
                st.error(f"Method Failed: {err}")
            else:
                # 1. Render Results (Math + Table + Eval)
                eval_pt = render_result(solver, df, expr, method, inverse=is_inverse)
                
                # 2. Render Graph
                render_interpolation_chart(solver, df_data, eval_pt, inverse=is_inverse, show_tangents=(method == "Hermite Interpolation"))
            