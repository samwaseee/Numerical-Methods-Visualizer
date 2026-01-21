import streamlit as st
import sympy as sp
import pandas as pd

def render_result(solver, df_table, expr, method_name, inverse=False):
    """
    Renders the mathematical result (LaTeX), the evaluation box, and the difference table.
    Returns the evaluated point tuple (input, output) or None.
    """
    # 1. Display Formula
    st.markdown("### Polynomial Model")
    
    # Convert expression to string for LaTeX
    # Note: The solver uses 'x' symbol. For inverse, we replace it visually.
    if expr is not None:
        poly_latex = sp.latex(sp.sympify(str(expr)))
        if inverse:
            poly_latex = poly_latex.replace("x", "y")
            st.info(f"**x = P(y):** ${poly_latex}$")
        else:
            st.info(f"**y = P(x):** ${poly_latex}$")
    elif "Spline" in method_name:
        st.info("**Piecewise Polynomials:** See table below for coefficients ($a + bx + cx^2 + dx^3$) per interval.")
    else:
        st.info("Result calculated.")

    # 2. Evaluation Section
    eval_result = None
    with st.container(border=True):
        st.markdown(f"**Evaluate {'X (given Y)' if inverse else 'Y (given X)'}**")
        
        c_input, c_result = st.columns([1, 2], gap="medium")
        
        input_label = "Enter y" if inverse else "Enter x"
        key_suffix = "inv" if inverse else "std"
        
        with c_input:
            val = st.number_input(
                input_label, 
                value=0.0, 
                step=0.1, 
                format="%.4f",
                key=f"eval_{key_suffix}_{method_name}"
            )
        
        with c_result:
            if solver.f:
                try:
                    res = float(solver.f(val))
                    label = f"x({val})" if inverse else f"f({val})"
                    st.metric(label=f"Result: {label}", value=f"{res:.6f}")
                    eval_result = (val, res)
                except Exception:
                    st.warning("Could not evaluate")
            else:
                st.caption("—")

    # 3. Table Expander
    if df_table is not None:
        table_label = "Show Coefficients" if "Spline" in method_name else ("Show Lagrange Steps" if "Lagrange" in method_name else "Show Difference Table")
        with st.expander(table_label, expanded=False):
            
            # --- FORMULA DISPLAY ---
            if "Lagrange" in method_name:
                in_var = "y" if inverse else "x"
                out_var = "x" if inverse else "y"
                st.markdown("#### Lagrange Basis Formula")
                st.latex(r"L_i(" + in_var + r") = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{" + in_var + r" - " + in_var + r"_j}{" + in_var + r"_i - " + in_var + r"_j}")
                st.latex(out_var + r"(" + in_var + r") = \sum_{i=0}^{n} " + out_var + r"_i L_i(" + in_var + r")")
            
            elif "Divided Difference" in method_name:
                st.markdown("#### Divided Difference Recursive Formula")
                st.latex(r"f[x_i, x_{i+1}] = \frac{y_{i+1} - y_i}{x_{i+1} - x_i}")
                st.latex(r"f[x_i, \dots, x_{i+k}] = \frac{f[x_{i+1}, \dots, x_{i+k}] - f[x_i, \dots, x_{i+k-1}]}{x_{i+k} - x_i}")
            
            elif "Newton Backward" in method_name:
                st.markdown("#### Backward Difference Formula")
                st.latex(r"\nabla y_i = y_i - y_{i-1}")
                st.latex(r"\nabla^k y_i = \nabla^{k-1} y_i - \nabla^{k-1} y_{i-1}")
            
            elif "Hermite" in method_name:
                st.markdown("#### Hermite Divided Difference")
                st.latex(r"z_{2i} = z_{2i+1} = x_i")
                st.latex(r"f[z_{2i}, z_{2i+1}] = y'_i")
                
            elif "Spline" in method_name:
                st.markdown("#### Cubic Spline Coefficients")
                st.latex(r"S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3")
            
            elif any(x in method_name for x in ["Gauss", "Stirling", "Bessel"]):
                st.markdown("#### Central Difference Formula ($\delta$)")
                st.latex(r"\delta y_i = y_{i+1} - y_i")
                st.latex(r"\delta^k y_i = \delta^{k-1} y_{i+1} - \delta^{k-1} y_i")

            else:
                # Newton Forward
                st.markdown("#### Forward Difference Formula ($\Delta$)")
                st.latex(r"\Delta y_i = y_{i+1} - y_i")
                st.latex(r"\Delta^k y_i = \Delta^{k-1} y_{i+1} - \Delta^{k-1} y_i")
            
            st.divider()
            
            # 1. Create a copy so we don't break the original data for math later
            df_display = df_table.copy()
            
            # 2. Identify which columns to format
            exclude_cols = ["Interval", "Term", "i"]
            target_cols = [c for c in df_display.columns if c not in exclude_cols]

            # 3. Apply formatting directly to the data
            for col in target_cols:
                # Convert the column to String/Object type
                df_display[col] = df_display[col].apply(
                    lambda x: "" if pd.isna(x) or str(x).strip() == "None" 
                    else "{:.4f}".format(float(x))
                )

            # 4. Display the Clean Dataframe
            st.dataframe(df_display, use_container_width=True)
            
    return eval_result