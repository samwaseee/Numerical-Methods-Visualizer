import streamlit as st
import pandas as pd
import numpy as np
from algorithms_fitting import FittingSolver
from components.fitting_graph import render_fitting_chart


def show_fitting_page():
    # --- NAVIGATION & CLEANUP ---
    if st.button("Back to Home", icon=":material/home:"):
        for key in list(st.session_state.keys()):
            if key != 'dark_mode':
                del st.session_state[key]
        st.query_params.clear()
        st.session_state.page = "landing"
        st.query_params["page"] = "landing"
        st.rerun()

    st.header("Curve Fitting (Regression)")
    st.markdown(
        "Fit a smooth curve to noisy data to model trends and make predictions."
    )

    # --- 1. INPUT & CONFIGURATION ---
    with st.container(border=True):
        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            st.subheader("1. Input Data")
            if "fit_data" not in st.session_state:
                st.session_state.fit_data = pd.DataFrame(
                    {
                        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        "y": [2.1, 3.9, 6.2, 8.1, 9.8, 12.2],
                    }
                )

            st.dataframe(
                st.session_state.fit_data, use_container_width=True, hide_index=True
            )

            c_add = st.columns(2)
            with c_add[0]:
                new_x = st.number_input("x", value=0.0, step=1.0, key="fit_new_x")
            with c_add[1]:
                new_y = st.number_input("y", value=0.0, step=1.0, key="fit_new_y")

            if st.button("Add Point", use_container_width=True):
                new_row = pd.DataFrame([{"x": new_x, "y": new_y}])
                st.session_state.fit_data = pd.concat(
                    [st.session_state.fit_data, new_row], ignore_index=True
                )
                st.session_state.fit_trigger = False
                st.rerun()

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Remove Last", use_container_width=True):
                    st.session_state.fit_data = st.session_state.fit_data.iloc[:-1]
                    st.session_state.fit_trigger = False
                    st.rerun()
            with b2:
                if st.button("Clear All", use_container_width=True):
                    st.session_state.fit_data = pd.DataFrame(columns=["x", "y"]).astype(
                        float
                    )
                    st.session_state.fit_trigger = False
                    st.rerun()

            x_vals = st.session_state.fit_data["x"].tolist()
            y_vals = st.session_state.fit_data["y"].tolist()

        with c2:
            st.subheader("2. Configuration")
            method = st.selectbox(
                "Fitting Method", ["Linear", "Polynomial", "Exponential"]
            )

            degree = 1
            if method == "Polynomial":
                max_deg = max(1, len(x_vals) - 1) if len(x_vals) > 1 else 10
                upper_bound = min(10, max_deg)
                degree = st.slider(
                    "Polynomial Degree", 1, upper_bound, min(2, upper_bound)
                )

            st.write("")
            if st.button("Fit Curve", type="primary", icon=":material/analytics:"):
                if len(x_vals) != len(y_vals) or len(x_vals) == 0:
                    st.error("Error: X and Y must have the same number of points (>0).")
                else:
                    st.session_state["fit_trigger"] = True
                    st.session_state["fit_params"] = (x_vals, y_vals, method, degree)

    # --- 2. RESULTS ---
    if st.session_state.get("fit_trigger"):
        x_data, y_data, method, degree = st.session_state["fit_params"]
        solver = FittingSolver(x_data, y_data)

        res = None
        if method == "Linear":
            res = solver.linear_regression()
        elif method == "Polynomial":
            res = solver.polynomial_regression(degree)
        elif method == "Exponential":
            res = solver.exponential_fit()

        if res and "error" in res:
            st.error(res["error"])
        elif res:
            st.divider()

            # --- CALCULATE ADVANCED METRICS ---
            n = len(x_data)
            
            # Determine parameters (k) and degree (p)
            if method == "Linear":
                p = 1
                k = 2
            elif method == "Polynomial":
                p = degree
                k = degree + 1
            elif method == "Exponential":
                p = 1 
                k = 2
            
            # Calculate Predictions & Residuals
            y_pred = [res["predict"](x) for x in x_data]
            residuals = [y - yp for y, yp in zip(y_data, y_pred)]
            sse = sum(r**2 for r in residuals)
            
            # 1. Adjusted R2
            if n > p + 1:
                r2_adj = 1 - (1 - res['r2']) * (n - 1) / (n - p - 1)
            else:
                r2_adj = np.nan
                
            # 2. Standard Error of Estimate (Se)
            if n > k:
                se = np.sqrt(sse / (n - k))
            else:
                se = 0.0

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("R² Score", f"{res['r2']:.4f}", help="Closer to 1.0 is better")
            with m2:
                if not np.isnan(r2_adj):
                    st.metric("Adjusted R²", f"{r2_adj:.4f}", help="The 'Overfitting Detector'. Penalizes complexity.")
                else:
                    st.metric("Adjusted R²", "N/A")
            with m3:
                st.metric("Std. Error (Se)", f"{se:.4f}", help="Average prediction error in Y units.")
            with m4:
                st.metric("MSE", f"{res['mse']:.4f}", help="Mean Squared Error")

            # Equation
            st.info(f"**Fitted Equation:** ${res['equation']}$")

            # 3. Residual Table
            with st.expander("Residual Analysis (Outlier Detection)", icon=":material/bar_chart:"):
                st.caption("Residual = Actual - Predicted. Large residuals indicate outliers.")
                df_res = pd.DataFrame({
                    "x": x_data,
                    "y (Actual)": y_data,
                    "y (Predicted)": y_pred,
                    "Residual": residuals
                })
                st.dataframe(
                    df_res.style.background_gradient(subset=["Residual"], cmap="coolwarm", vmin=-se*2, vmax=se*2),
                    use_container_width=True
                )

            # --- STEP-BY-STEP CALCULATION ---
            with st.expander("Show Calculation Steps"):
                steps = res.get("steps", {})

                if method == "Linear":
                    st.markdown("### Linear Regression (Least Squares)")
                    st.markdown(
                        "We solve the system of Normal Equations for $y = a_0 + a_1 x$:"
                    )
                    st.latex(
                        r"""
                    \begin{cases}
                    n a_0 + (\sum x) a_1 = \sum y \\
                    (\sum x) a_0 + (\sum x^2) a_1 = \sum xy
                    \end{cases}
                    """
                    )

                    st.markdown("**1. Calculate Sums:**")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("n", steps["n"])
                    c2.metric("Σx", f"{steps['sx']:.4f}")
                    c3.metric("Σy", f"{steps['sy']:.4f}")
                    c4.metric("Σx²", f"{steps['sxx']:.4f}")
                    c5.metric("Σxy", f"{steps['sxy']:.4f}")

                    st.markdown("**2. Substitute into System:**")
                    st.latex(
                        f"""
                    \\begin{{bmatrix}}
                    {steps['n']} & {steps['sx']:.4f} \\\\
                    {steps['sx']:.4f} & {steps['sxx']:.4f}
                    \\end{{bmatrix}}
                    \\begin{{bmatrix}} a_0 \\\\ a_1 \\end{{bmatrix}}
                    =
                    \\begin{{bmatrix}} {steps['sy']:.4f} \\\\ {steps['sxy']:.4f} \\end{{bmatrix}}
                    """
                    )

                    st.markdown("**3. Solve for Coefficients:**")
                    st.latex(f"a_0 = {steps['a0']:.4f}, \\quad a_1 = {steps['a1']:.4f}")

                elif method == "Polynomial":
                    deg = steps["degree"]
                    st.markdown(f"### Polynomial Regression (Degree {deg})")
                    st.markdown(
                        f"We solve the matrix equation $X^T X A = X^T Y$ for $y = a_0 + a_1 x + \dots + a_k x^k$."
                    )

                    st.markdown("**Normal Equation Matrix:**")

                    # Construct LaTeX matrix
                    mat = steps["matrix"]
                    rhs = steps["rhs"]

                    rows = []
                    for i in range(len(mat)):
                        row_str = " & ".join([f"{x:.2f}" for x in mat[i]])
                        rows.append(row_str)
                    mat_latex = "\\\\ \n".join(rows)
                    rhs_latex = "\\\\ \n".join([f"{x:.2f}" for x in rhs])
                    coeffs_latex = "\\\\ \n".join([f"a_{i}" for i in range(len(mat))])

                    st.latex(
                        f"""
                    \\begin{{bmatrix}}
                    {mat_latex}
                    \\end{{bmatrix}}
                    \\begin{{bmatrix}} {coeffs_latex} \\end{{bmatrix}}
                    =
                    \\begin{{bmatrix}} {rhs_latex} \\end{{bmatrix}}
                    """
                    )

                    st.markdown("**Solved Coefficients:**")
                    coeffs = res["coeffs"]  # [ak, ..., a0]
                    coeffs_rev = coeffs[::-1]  # [a0, ..., ak]

                    c_cols = st.columns(min(len(coeffs), 4))
                    for i, c in enumerate(coeffs_rev):
                        c_cols[i % 4].metric(f"a_{i}", f"{c:.4f}")

                elif method == "Exponential":
                    st.markdown("### Exponential Fit (Linearization)")
                    st.markdown("Model: $y = a e^{bx}$")
                    st.markdown(
                        "Linearize by taking natural log: $\\ln(y) = \\ln(a) + bx$"
                    )
                    st.markdown(
                        "Let $Y = \\ln(y)$, $A = \\ln(a)$, $B = b$. Fit $Y = A + Bx$."
                    )

                    st.markdown("**1. Transformed Sums:**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Σx", f"{steps['sx']:.4f}")
                    c2.metric("Σln(y)", f"{steps['sy_log']:.4f}")
                    c3.metric("Σx²", f"{steps['sxx']:.4f}")
                    c4.metric("Σx·ln(y)", f"{steps['sxy_log']:.4f}")

                    st.markdown("**2. Solve Linear System for A and B:**")
                    st.latex(
                        f"""
                    \\begin{{bmatrix}}
                    {steps['n']} & {steps['sx']:.4f} \\\\
                    {steps['sx']:.4f} & {steps['sxx']:.4f}
                    \\end{{bmatrix}}
                    \\begin{{bmatrix}} A \\\\ B \\end{{bmatrix}}
                    =
                    \\begin{{bmatrix}} {steps['sy_log']:.4f} \\\\ {steps['sxy_log']:.4f} \\end{{bmatrix}}
                    """
                    )

                    st.markdown("**3. Resulting Coefficients:**")
                    c_res1, c_res2 = st.columns(2)
                    with c_res1:
                        st.latex(
                            f"A = {steps['A']:.4f} \\implies a = e^A = {steps['a']:.4f}"
                        )
                    with c_res2:
                        st.latex(f"B = {steps['B']:.4f} \\implies b = {steps['b']:.4f}")

            # Visualization
            render_fitting_chart(x_data, y_data, res)

            # Prediction Tool
            with st.container(border=True):
                # Professional SVG Icon for Prediction
                st.markdown("""
                    <h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <svg xmlns="http://www.w3.org/2000/svg" height="28px" viewBox="0 -960 960 960" width="28px" fill="#A855F7"><path d="M480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320q-33 0-56.5-23.5T400-480q0-33 23.5-56.5T480-560q33 0 56.5 23.5T560-480q0 33-23.5 56.5T480-480Z"/></svg>
                        Prediction
                    </h3>
                """, unsafe_allow_html=True)
                c_pred1, c_pred2 = st.columns([1, 2])
                with c_pred1:
                    val_x = st.number_input(
                        "Predict Y at x =", value=float(x_data[-1]) + 1.0
                    )
                with c_pred2:
                    val_y = res["predict"](val_x)
                    st.metric("Predicted Y", f"{val_y:.4f}")
