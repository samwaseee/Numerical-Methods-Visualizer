import streamlit as st
import pandas as pd
import sympy as sp
import numpy as np

def render_differentiation_results(res, df_table, method, orig_data, target_x, extrema=None):
    """
    Renders the differentiation results: Metrics, Formulas, and Tables.
    """
    # --- 1. SUMMARY METRICS ---
    with st.container(border=True):
        st.markdown(f"#### Results at $x = {target_x}$")
        m1, m2, m3 = st.columns(3)
        with m1: 
            st.metric("f(x)", f"{res['y']:.4f}")
        with m2: 
            st.metric("f'(x)", f"{res['dy']:.4f}")
        with m3: 
            st.metric("f''(x)", f"{res['d2y']:.4f}")

    # --- NEW: GROUND TRUTH COMPARISON ---
    if "true_func_str" in st.session_state and st.session_state.true_func_str:
        try:
            func_str = st.session_state.true_func_str
            x_sym = sp.symbols('x')
            # Parse
            expr = sp.sympify(func_str.replace("^", "**"))
            
            # Differentiate
            d1_expr = sp.diff(expr, x_sym, 1)
            d2_expr = sp.diff(expr, x_sym, 2)
            
            # Evaluate
            true_dy = float(d1_expr.subs(x_sym, target_x))
            true_d2y = float(d2_expr.subs(x_sym, target_x))
            
            # Errors
            err_dy = abs(true_dy - res['dy'])
            err_d2y = abs(true_d2y - res['d2y'])
            
            # Relative Errors (handle div by zero)
            rel_dy = (err_dy / abs(true_dy) * 100) if abs(true_dy) > 1e-9 else 0.0
            rel_d2y = (err_d2y / abs(true_d2y) * 100) if abs(true_d2y) > 1e-9 else 0.0
            
            with st.container(border=True):
                st.markdown(f"#### 🔍 Ground Truth Comparison")
                st.caption(f"Based on generated function: $f(x) = {sp.latex(expr)}$")
                
                c_gt1, c_gt2 = st.columns(2)
                with c_gt1:
                    st.markdown("**1st Derivative ($f'$)**")
                    st.write(f"True Value: `{true_dy:.6f}`")
                    st.write(f"Abs Error: `{err_dy:.6e}`")
                    st.write(f"Rel Error: `{rel_dy:.4f}%`")
                    
                with c_gt2:
                    st.markdown("**2nd Derivative ($f''$)**")
                    st.write(f"True Value: `{true_d2y:.6f}`")
                    st.write(f"Abs Error: `{err_d2y:.6e}`")
                    st.write(f"Rel Error: `{rel_d2y:.4f}%`")
                    
        except Exception as e:
            pass

    # --- OPTIMIZATION RESULTS (Minima/Maxima) ---
    if extrema:
        st.subheader("Optimization Results (Minima / Maxima)")
        if len(extrema) == 0:
            st.info("No local extrema found within the data range.")
        else:
            cols = st.columns(min(len(extrema), 4))
            for i, pt in enumerate(extrema):
                with cols[i % 4]:
                    st.metric(
                        label=f"{pt['type']}",
                        value=f"x = {pt['x']:.4f}",
                        delta=f"y = {pt['y']:.4f}",
                        delta_color="off"
                    )

    # --- EXPANDER FOR DETAILED ANALYSIS ---
    with st.expander("Show Step-by-Step Analysis", icon=":material/analytics:"):
        
        # --- 1. POLYNOMIAL MODEL (MOVED) ---
        with st.container(border=True):
            st.markdown("##### 📝 Interpolating Polynomial")
            poly_expr = res.get('expr')

            if "Least Squares" in method:
                A, B, C = res['coeffs']
                st.info(f"Fitted Model: $y = {A:.4f}x^2 + {B:.4f}x + {C:.4f}$")
            elif "Spline" in method:
                st.info("Using Piecewise Cubic Splines. See coefficients in the table below.")
            elif poly_expr:
                try:
                    poly_latex = sp.latex(sp.sympify(str(poly_expr).replace('^', '**')))
                    st.info(f"$P(x) = {poly_latex}$")
                except:
                    st.code(f"P(x) = {poly_expr}", language="python")
            else:
                st.caption("No single polynomial expression available for this method.")

        # --- 2. DEEP DIVE: FORMULA & SUBSTITUTION ---
        st.markdown("### 📘 Step-by-Step Substitution")

        def get_dynamic_latex(method_type, order, max_k):
            u_sym = sp.symbols('u')
            terms = []
            limit = max_k
            
            for k in range(order, limit + 1):
                term_poly = 1
                sym_diff = ""
                
                if method_type == "Forward":
                    for i in range(k): term_poly *= (u_sym - i)
                    sym_diff = f"\\Delta^{{{k}}} y_0" if k > 1 else "\\Delta y_0"
                elif method_type == "Backward":
                    for i in range(k): term_poly *= (u_sym + i)
                    sym_diff = f"\\nabla^{{{k}}} y_n" if k > 1 else "\\nabla y_n"
                elif method_type == "Central":
                    # Gauss Forward (Standard Central Difference)
                    for i in range(k):
                        if i == 0: factor = 0
                        elif i % 2 == 1: factor = -((i + 1) // 2)
                        else: factor = (i // 2)
                        term_poly *= (u_sym + factor)
                    sym_diff = f"\\delta^{{{k}}} y_{{{-(k//2)}}}"
                elif method_type == "Stirling":
                    # Stirling's Formula
                    if k % 2 != 0: # Odd
                        term_poly = u_sym
                        for m in range(1, (k - 1) // 2 + 1):
                            term_poly *= (u_sym**2 - m**2)
                        sym_diff = f"\\mu \\delta^{{{k}}} y_0" if k > 1 else "\\mu \\delta y_0"
                    else: # Even
                        term_poly = u_sym**2
                        for m in range(1, (k - 2) // 2 + 1):
                            term_poly *= (u_sym**2 - m**2)
                        sym_diff = f"\\delta^{{{k}}} y_0"
                elif method_type == "Bessel":
                    # Bessel's Formula
                    if k == 1:
                        term_poly = u_sym - 0.5
                        sym_diff = "\\delta y_{1/2}"
                    elif k % 2 == 0: # Even
                        term_poly = 1
                        for m in range(k // 2):
                            term_poly *= (u_sym + m) * (u_sym - (m + 1))
                        sym_diff = f"\\mu \\delta^{{{k}}} y_{{1/2}}"
                    else: # Odd > 1
                        term_poly = u_sym - 0.5
                        for m in range((k - 1) // 2):
                            term_poly *= (u_sym + m) * (u_sym - (m + 1))
                        sym_diff = f"\\delta^{{{k}}} y_{{1/2}}"
                
                term_poly = term_poly / sp.factorial(k)
                coeff = sp.diff(term_poly, u_sym, order)
                
                c_latex = sp.latex(sp.simplify(coeff))
                if c_latex == "0": continue
                
                if c_latex == "1": terms.append(sym_diff)
                elif c_latex == "-1": terms.append(f"- {sym_diff}")
                else: terms.append(f"\\left( {c_latex} \\right) {sym_diff}")
                
            if max_k > limit: terms.append("\\dots")
            body = " + ".join(terms).replace("+ -", "- ")
            h_part = f"h^{order}" if order > 1 else "h"
            return f"f{''.join(['\'' for _ in range(order)])}(x) \\approx \\frac{{1}}{{{h_part}}} \\left[ {body} \\right]"

        # Helper: Calculate h and u for display
        h_val = 1.0
        if len(orig_data) > 1:
            h_val = orig_data['x'].iloc[1] - orig_data['x'].iloc[0]

        # --- NEWTON FORWARD INTERPRETATION ---
        if "Newton Forward" in method:
            x0 = orig_data['x'].iloc[0]
            u_val = (target_x - x0) / h_val
            u_sym = sp.symbols('u')
            max_k = len(df_table.columns) - 2

            tab1, tab2 = st.tabs(["1st Derivative Formula", "2nd Derivative Formula"])
            
            with tab1:
                st.markdown("**1. The General Formula (Forward $\Delta$):**")
                st.latex(get_dynamic_latex("Forward", 1, max_k))
                
                st.markdown(f"**2. Substitution ($h={h_val:.4f}, u={u_val:.4f}$):**")
                
                latex_parts = []
                for k in range(1, max_k + 1):
                    val = df_table.iloc[0, k+1]
                    if pd.isna(val): continue
                    
                    # Construct polynomial explicitly to avoid polygamma
                    term_poly = 1
                    for i in range(k): term_poly *= (u_sym - i)
                    term_poly = term_poly / sp.factorial(k)
                    coeff_poly = sp.diff(term_poly, u_sym)

                    if k == 1:
                        latex_parts.append(f"({val:.4f})")
                    else:
                        c_latex = sp.latex(sp.simplify(coeff_poly))
                        c_sub = c_latex.replace("u", f"({u_val:.2f})")
                        latex_parts.append(f"\\left[ {c_sub} \\right]({val:.4f})")
                        
                full_latex = f"f'({target_x}) \\approx \\frac{{1}}{{{h_val}}} \\left[ " + " + ".join(latex_parts) + " \\right]"
                st.latex(full_latex)
                st.success(f"**Result:** {res['dy']:.6f}")

            with tab2:
                st.markdown("**1. The General Formula (Forward $\Delta$):**")
                st.latex(get_dynamic_latex("Forward", 2, max_k))
                
                st.markdown(f"**2. Substitution:**")
                
                latex_parts_2 = []
                for k in range(2, max_k + 1):
                    val = df_table.iloc[0, k+1]
                    if pd.isna(val): continue
                    
                    # Construct polynomial explicitly
                    term_poly = 1
                    for i in range(k): term_poly *= (u_sym - i)
                    term_poly = term_poly / sp.factorial(k)
                    coeff_poly = sp.diff(term_poly, u_sym, 2)

                    if k == 2:
                        latex_parts_2.append(f"({val:.4f})")
                    else:
                        c_latex = sp.latex(sp.simplify(coeff_poly))
                        c_sub = c_latex.replace("u", f"({u_val:.2f})")
                        latex_parts_2.append(f"\\left[ {c_sub} \\right]({val:.4f})")
                        
                if latex_parts_2:
                    full_latex_2 = f"f''({target_x}) \\approx \\frac{{1}}{{{h_val}^2}} \\left[ " + " + ".join(latex_parts_2) + " \\right]"
                    st.latex(full_latex_2)
                    st.success(f"**Result:** {res['d2y']:.6f}")
                else:
                    st.warning("Not enough points for 2nd derivative.")

        # --- NEWTON BACKWARD INTERPRETATION ---
        elif "Newton Backward" in method:
            xn = orig_data['x'].iloc[-1]
            u_val = (target_x - xn) / h_val
            u_sym = sp.symbols('u')
            max_k = len(df_table.columns) - 2

            tab1, tab2 = st.tabs(["1st Derivative Formula", "2nd Derivative Formula"])
            
            with tab1:
                st.markdown("**1. The General Formula (Backward):**")
                st.latex(get_dynamic_latex("Backward", 1, max_k))
                
                st.markdown(f"**2. Substitution ($h={h_val:.4f}, u={u_val:.4f}$):**")
                
                latex_parts = []
                for k in range(1, max_k + 1):
                    row_idx = len(df_table) - 1
                    val = df_table.iloc[row_idx, k+1]
                    if pd.isna(val): continue
                    
                    # Construct polynomial explicitly: (u+k-1)...(u) / k!
                    term_poly = 1
                    for i in range(k): term_poly *= (u_sym + i)
                    term_poly = term_poly / sp.factorial(k)
                    coeff_poly = sp.diff(term_poly, u_sym)

                    if k == 1:
                        latex_parts.append(f"({val:.4f})")
                    else:
                        c_latex = sp.latex(sp.simplify(coeff_poly))
                        c_sub = c_latex.replace("u", f"({u_val:.2f})")
                        latex_parts.append(f"\\left[ {c_sub} \\right]({val:.4f})")
                        
                full_latex = f"f'({target_x}) \\approx \\frac{{1}}{{{h_val}}} \\left[ " + " + ".join(latex_parts) + " \\right]"
                st.latex(full_latex)
                st.success(f"**Result:** {res['dy']:.6f}")

            with tab2:
                st.markdown("**1. The General Formula (Backward):**")
                st.latex(get_dynamic_latex("Backward", 2, max_k))
                
                st.markdown(f"**2. Substitution:**")
                
                latex_parts_2 = []
                for k in range(2, max_k + 1):
                    row_idx = len(df_table) - 1
                    val = df_table.iloc[row_idx, k+1]
                    if pd.isna(val): continue
                    
                    # Construct polynomial explicitly
                    term_poly = 1
                    for i in range(k): term_poly *= (u_sym + i)
                    term_poly = term_poly / sp.factorial(k)
                    coeff_poly = sp.diff(term_poly, u_sym, 2)

                    if k == 2:
                        latex_parts_2.append(f"({val:.4f})")
                    else:
                        c_latex = sp.latex(sp.simplify(coeff_poly))
                        c_sub = c_latex.replace("u", f"({u_val:.2f})")
                        latex_parts_2.append(f"\\left[ {c_sub} \\right]({val:.4f})")
                        
                if latex_parts_2:
                    full_latex_2 = f"f''({target_x}) \\approx \\frac{{1}}{{{h_val}^2}} \\left[ " + " + ".join(latex_parts_2) + " \\right]"
                    st.latex(full_latex_2)
                    st.success(f"**Result:** {res['d2y']:.6f}")
                else:
                    st.warning("Not enough points for 2nd derivative.")

        # --- STIRLING'S FORMULA INTERPRETATION ---
        elif "Stirling" in method:
            # Find closest point x0 (Center)
            x_vals = orig_data['x'].values
            idx0 = (np.abs(x_vals - target_x)).argmin()
            x0 = x_vals[idx0]
            
            u_val = (target_x - x0) / h_val
            u_sym = sp.symbols('u')
            max_k = len(df_table.columns) - 2
            
            def get_stirling_term(k, u_sym):
                if k % 2 != 0: # Odd
                    term = u_sym
                    for m in range(1, (k - 1) // 2 + 1): term *= (u_sym**2 - m**2)
                else: # Even
                    term = u_sym**2
                    for m in range(1, (k - 2) // 2 + 1): term *= (u_sym**2 - m**2)
                return term / sp.factorial(k)

            def get_stirling_val(k, idx0, df_table):
                # Returns value or average value for Stirling
                try:
                    if k % 2 != 0: # Odd (Average)
                        if idx0 - 1 < 0: return None
        
                        v1 = df_table.iloc[idx0, k+1]
                        v2 = df_table.iloc[idx0 - 1, k+1]
                        if pd.isna(v1) or pd.isna(v2): return None
                        return (v1 + v2) / 2
                    else: # Even
                        # Shift k//2. Row = idx0.
                        val = df_table.iloc[idx0, k+1]
                        if pd.isna(val): return None
                        return val
                except: return None
                
            # Calculate how deep we can go for THIS specific row
            dist_up = idx0 
            dist_down = (len(df_table) - 1) - idx0
            safe_radius = min(dist_up, dist_down)
            
            # Central differences generally consume 1 radius unit for every 2 orders
            # So max order is roughly 2 * radius
            valid_max_k = safe_radius * 2
            valid_k = min(max_k, valid_max_k)

            tab1, tab2 = st.tabs(["1st Derivative Formula", "2nd Derivative Formula"])

            with tab1:
                st.markdown("**1. The General Formula (Stirling):**")
                st.latex(get_dynamic_latex("Stirling", 1, valid_k))
                if valid_k < max_k:
                    st.caption(f"Note: Terms above order {valid_k} are unavailable near the data boundary.")
                
                st.markdown(f"**2. Substitution ($h={h_val:.4f}, u={u_val:.4f}$):**")
                
                latex_parts = []
                for k in range(1, valid_k + 1):
                    val = get_stirling_val(k, idx0, df_table)
                    if val is None: continue
                    
                    term_poly = get_stirling_term(k, u_sym)
                    coeff_poly = sp.diff(term_poly, u_sym, 1)

                    c_latex = sp.latex(sp.simplify(coeff_poly))
                    c_sub = c_latex.replace("u", f"({u_val:.2f})")
                    
                    if c_latex == "1": latex_parts.append(f"({val:.4f})")
                    else: latex_parts.append(f"\\left[ {c_sub} \\right]({val:.4f})")
                        
                full_latex = f"f'({target_x}) \\approx \\frac{{1}}{{{h_val}}} \\left[ " + " + ".join(latex_parts) + " \\right]"
                st.latex(full_latex)
                st.success(f"**Result:** {res['dy']:.6f}")

            with tab2:
                st.markdown("**1. The General Formula (Stirling):**")
                st.latex(get_dynamic_latex("Stirling", 2, valid_k))
                
                st.markdown(f"**2. Substitution:**")
                
                latex_parts_2 = []
                for k in range(2, valid_k + 1):
                    val = get_stirling_val(k, idx0, df_table)
                    if val is None: continue
                    
                    term_poly = get_stirling_term(k, u_sym)
                    coeff_poly = sp.diff(term_poly, u_sym, 2)

                    c_latex = sp.latex(sp.simplify(coeff_poly))
                    c_sub = c_latex.replace("u", f"({u_val:.2f})")
                    
                    if c_latex == "1": latex_parts_2.append(f"({val:.4f})")
                    else: latex_parts_2.append(f"\\left[ {c_sub} \\right]({val:.4f})")
                        
                if latex_parts_2:
                    full_latex_2 = f"f''({target_x}) \\approx \\frac{{1}}{{{h_val}^2}} \\left[ " + " + ".join(latex_parts_2) + " \\right]"
                    st.latex(full_latex_2)
                    st.success(f"**Result:** {res['d2y']:.6f}")
                else:
                    st.warning("Not enough points for 2nd derivative.")

        # --- BESSEL'S FORMULA INTERPRETATION ---
        elif "Bessel" in method:
            # Find mid point x0 (Left of center for u ~ 0.5)
            x_vals = orig_data['x'].values
            mid = (len(x_vals) - 1) // 2
            x0 = x_vals[mid]
            
            u_val = (target_x - x0) / h_val
            u_sym = sp.symbols('u')
            max_k = len(df_table.columns) - 2
            
            def get_bessel_term(k, u_sym):
                if k == 1:
                    term = u_sym - 0.5
                elif k % 2 == 0: # Even
                    term = 1
                    for m in range(k // 2): term *= (u_sym + m) * (u_sym - (m + 1))
                else: # Odd > 1
                    term = u_sym - 0.5
                    for m in range((k - 1) // 2): term *= (u_sym + m) * (u_sym - (m + 1))
                return term / sp.factorial(k)

            def get_bessel_val(k, mid, df_table):
                try:
                    if k == 1:
                        return df_table.iloc[mid, 1+1] # Delta y0 (Shift 0 -> Row mid)
                    elif k % 2 == 0: # Even (Average)
                        # Shift k//2.
                        # Term 1: mid - k//2. Row = mid.
                        # Term 2: mid - k//2 + 1. Row = mid + 1.
                        v1 = df_table.iloc[mid, k+1]
                        v2 = df_table.iloc[mid + 1, k+1]
                        if pd.isna(v1) or pd.isna(v2): return None
                        return (v1 + v2) / 2
                    else: # Odd > 1
                        # Shift k//2 = (k-1)//2.
                        # Term: mid - (k-1)//2. Row = mid.
                        val = df_table.iloc[mid, k+1]
                        if pd.isna(val): return None
                        return val
                except: return None
                
            # Determine valid max order
            valid_k = 0
            for k in range(1, max_k + 1):
                if get_bessel_val(k, mid, df_table) is None: break
                valid_k = k

            tab1, tab2 = st.tabs(["1st Derivative Formula", "2nd Derivative Formula"])

            with tab1:
                st.markdown("**1. The General Formula (Bessel):**")
                st.latex(get_dynamic_latex("Bessel", 1, valid_k))
                if valid_k < max_k:
                    st.caption(f"Note: Terms above order {valid_k} are unavailable near the data boundary.")
                
                st.markdown(f"**2. Substitution ($h={h_val:.4f}, u={u_val:.4f}$):**")
                
                latex_parts = []
                for k in range(1, valid_k + 1):
                    val = get_bessel_val(k, mid, df_table)
                    if val is None: continue
                    
                    term_poly = get_bessel_term(k, u_sym)
                    coeff_poly = sp.diff(term_poly, u_sym, 1)

                    c_latex = sp.latex(sp.simplify(coeff_poly))
                    c_sub = c_latex.replace("u", f"({u_val:.2f})")
                    
                    if c_latex == "1": latex_parts.append(f"({val:.4f})")
                    else: latex_parts.append(f"\\left[ {c_sub} \\right]({val:.4f})")
                
                full_latex = f"f'({target_x}) \\approx \\frac{{1}}{{{h_val}}} \\left[ " + " + ".join(latex_parts) + " \\right]"
                st.latex(full_latex)
                st.success(f"**Result:** {res['dy']:.6f}")

            with tab2:
                st.markdown("**1. The General Formula (Bessel):**")
                st.latex(get_dynamic_latex("Bessel", 2, valid_k))
                
                st.markdown(f"**2. Substitution:**")
                latex_parts_2 = []
                for k in range(2, valid_k + 1):
                    val = get_bessel_val(k, mid, df_table)
                    if val is None: continue
                    term_poly = get_bessel_term(k, u_sym)
                    coeff_poly = sp.diff(term_poly, u_sym, 2)
                    c_latex = sp.latex(sp.simplify(coeff_poly))
                    c_sub = c_latex.replace("u", f"({u_val:.2f})")
                    if c_latex == "1": latex_parts_2.append(f"({val:.4f})")
                    else: latex_parts_2.append(f"\\left[ {c_sub} \\right]({val:.4f})")
                if latex_parts_2:
                    full_latex_2 = f"f''({target_x}) \\approx \\frac{{1}}{{{h_val}^2}} \\left[ " + " + ".join(latex_parts_2) + " \\right]"
                    st.latex(full_latex_2)
                    st.success(f"**Result:** {res['d2y']:.6f}")
                else:
                    st.warning("Not enough points for 2nd derivative.")

        # --- LAGRANGE INTERPRETATION ---
        elif "Lagrange" in method:
            x_vals = orig_data['x'].values
            y_vals = orig_data['y'].values
            n = len(x_vals)
            x_sym = sp.symbols('x')
            
            # Pre-calculate Basis polynomials L_i(x) for symbolic differentiation
            L_exprs = []
            for i in range(n):
                num = 1
                den = 1
                for j in range(n):
                    if i != j:
                        num *= (x_sym - x_vals[j])
                        den *= (x_vals[i] - x_vals[j])
                L_exprs.append(num/den)

            tab1, tab2 = st.tabs(["1st Derivative Formula", "2nd Derivative Formula"])
            
            with tab1:
                st.markdown("**1. The General Formula (Lagrange):**")
                st.markdown("The derivative is the sum of weighted basis derivatives:")
                
                # Construct formula string: y_0 L'_0(x) + ...
                terms_latex = []
                for i in range(n):
                    terms_latex.append(f"y_{{{i}}} L'_{{{i}}}(x)")
                
                if n > 5:
                    formula_latex = " + ".join(terms_latex[:3]) + " + \\dots + " + terms_latex[-1]
                else:
                    formula_latex = " + ".join(terms_latex)
                
                st.latex(f"f'(x) \\approx {formula_latex}")
                
                st.markdown(f"**2. Substitution ($x={target_x}$):**")
                
                sub_parts = []
                for i in range(n):
                    deriv = sp.diff(L_exprs[i], x_sym, 1)
                    val_deriv = float(deriv.subs(x_sym, target_x))
                    # Format: (y_i)(L'_i(x))
                    sub_parts.append(f"({y_vals[i]:.4f})({val_deriv:.4f})")
                
                if n > 5:
                    sub_latex = " + ".join(sub_parts[:3]) + " + \\dots + " + sub_parts[-1]
                else:
                    sub_latex = " + ".join(sub_parts)
                
                st.latex(f"f'({target_x}) \\approx {sub_latex}")
                st.success(f"**Result:** {res['dy']:.6f}")

            with tab2:
                st.markdown("**1. The General Formula (Lagrange):**")
                terms_latex = []
                for i in range(n):
                    terms_latex.append(f"y_{{{i}}} L''_{{{i}}}(x)")
                
                if n > 5:
                    formula_latex = " + ".join(terms_latex[:3]) + " + \\dots + " + terms_latex[-1]
                else:
                    formula_latex = " + ".join(terms_latex)
                
                st.latex(f"f''(x) \\approx {formula_latex}")
                
                st.markdown(f"**2. Substitution ($x={target_x}$):**")
                
                sub_parts = []
                for i in range(n):
                    deriv2 = sp.diff(L_exprs[i], x_sym, 2)
                    val_deriv2 = float(deriv2.subs(x_sym, target_x))
                    sub_parts.append(f"({y_vals[i]:.4f})({val_deriv2:.4f})")
                
                if n > 5:
                    sub_latex = " + ".join(sub_parts[:3]) + " + \\dots + " + sub_parts[-1]
                else:
                    sub_latex = " + ".join(sub_parts)
                    
                st.latex(f"f''({target_x}) \\approx {sub_latex}")
                st.success(f"**Result:** {res['d2y']:.6f}")

        # --- NEWTON DIVIDED DIFFERENCE INTERPRETATION ---
        elif "Divided Difference" in method:
            x_vals = orig_data['x'].values
            n = len(x_vals)
            x_sym = sp.symbols('x')

            tab1, tab2 = st.tabs(["1st Derivative Formula", "2nd Derivative Formula"])
            
            with tab1:
                st.markdown("**1. The General Formula (Newton Divided Difference):**")
                st.markdown("Differentiating the Newton polynomial term by term:")
                
                latex_terms = []
                # k=1 to n-1
                for k in range(1, n):
                    coeff_sym = f"f[x_0, \\dots, x_{{{k}}}]"
                    
                    # Product term derivative representation
                    if k == 1:
                        prod_term = "1"
                    elif k == 2:
                        prod_term = "(2x - x_0 - x_1)"
                    else:
                        prod_term = f"\\frac{{d}}{{dx}} \\prod_{{j=0}}^{{{k-1}}} (x - x_j)"
                    
                    if prod_term == "1": latex_terms.append(coeff_sym)
                    else: latex_terms.append(f"{coeff_sym} \\cdot {prod_term}")
                
                if len(latex_terms) > 4:
                    f_latex = " + ".join(latex_terms[:3]) + " + \\dots + " + latex_terms[-1]
                else:
                    f_latex = " + ".join(latex_terms)
                
                st.latex(f"f'(x) \\approx {f_latex}")
                
                st.markdown(f"**2. Substitution ($x={target_x}$):**")
                
                sub_parts = []
                for k in range(1, n):
                    # Coefficient from table (Row 0, Col k+1 because of 'x' col)
                    b_k = df_table.iloc[0, k+1]
                    if pd.isna(b_k): continue
                    
                    # Calculate derivative of product term
                    term_poly = 1
                    for j in range(k): term_poly *= (x_sym - x_vals[j])
                    
                    deriv_poly = sp.diff(term_poly, x_sym, 1)
                    val_deriv = float(deriv_poly.subs(x_sym, target_x))
                    
                    sub_parts.append(f"({b_k:.4f})({val_deriv:.4f})")
                
                if len(sub_parts) > 4:
                    sub_latex = " + ".join(sub_parts[:3]) + " + \\dots + " + sub_parts[-1]
                else:
                    sub_latex = " + ".join(sub_parts)
                
                st.latex(f"f'({target_x}) \\approx {sub_latex}")
                st.success(f"**Result:** {res['dy']:.6f}")

            with tab2:
                st.markdown("**1. The General Formula (Newton DD):**")
                
                latex_terms = []
                for k in range(2, n):
                    coeff_sym = f"f[x_0, \\dots, x_{{{k}}}]"
                    prod_term = f"\\frac{{d^2}}{{dx^2}} \\prod_{{j=0}}^{{{k-1}}} (x - x_j)"
                    latex_terms.append(f"{coeff_sym} \\cdot {prod_term}")
                
                if not latex_terms:
                    st.info("Higher order terms needed.")
                else:
                    if len(latex_terms) > 4:
                        f_latex = " + ".join(latex_terms[:3]) + " + \\dots + " + latex_terms[-1]
                    else:
                        f_latex = " + ".join(latex_terms)
                    st.latex(f"f''(x) \\approx {f_latex}")
                
                st.markdown(f"**2. Substitution ($x={target_x}$):**")
                
                sub_parts = []
                for k in range(2, n):
                    b_k = df_table.iloc[0, k+1]
                    if pd.isna(b_k): continue
                    
                    term_poly = 1
                    for j in range(k): term_poly *= (x_sym - x_vals[j])
                    
                    deriv2_poly = sp.diff(term_poly, x_sym, 2)
                    val_deriv2 = float(deriv2_poly.subs(x_sym, target_x))
                    
                    sub_parts.append(f"({b_k:.4f})({val_deriv2:.4f})")
                
                if sub_parts:
                    if len(sub_parts) > 4:
                        sub_latex = " + ".join(sub_parts[:3]) + " + \\dots + " + sub_parts[-1]
                    else:
                        sub_latex = " + ".join(sub_parts)
                    st.latex(f"f''({target_x}) \\approx {sub_latex}")
                    st.success(f"**Result:** {res['d2y']:.6f}")
                else:
                    st.warning("Not enough points for 2nd derivative.")

        # --- RICHARDSON EXTRAPOLATION ---
        elif "Richardson" in method:
            st.markdown("### 🚀 Richardson Extrapolation")
            st.markdown("Combines two lower-accuracy estimates to create a higher-accuracy one.")
            
            tab1, tab2 = st.tabs(["1st Derivative Formula", "2nd Derivative Formula"])
            
            with tab1:
                st.markdown("**1. The General Formula:**")
                st.latex(r"D \approx \frac{4 D(h) - D(2h)}{3}")
                st.markdown(f"**2. Substitution:**")
                st.markdown(f"""
                * $D(h) = {res['D_h']:.6f}$ (Step $h={res['h']:.4f}$)
                * $D(2h) = {res['D_2h']:.6f}$ (Step $2h={2*res['h']:.4f}$)
                """)
                st.success(f"**Result:** {res['dy']:.6f}")
                
            with tab2:
                st.markdown("**1. The General Formula:**")
                st.latex(r"D'' \approx \frac{4 D''(h) - D''(2h)}{3}")
                st.markdown(f"**2. Substitution:**")
                st.markdown(f"""
                * $D''(h) = {res['D2_h']:.6f}$
                * $D''(2h) = {res['D2_2h']:.6f}$
                """)
                st.success(f"**Result:** {res['d2y']:.6f}")

        # --- CUBIC SPLINE ---
        elif "Spline" in method:
            st.markdown("### 〰️ Cubic Spline Differentiation")
            c = res['coeffs']
            dx = res['dx']
            
            st.markdown(f"**Active Interval:** {c['Interval']}")
            st.latex(r"S(x) = a + b(x-x_i) + c(x-x_i)^2 + d(x-x_i)^3")
            st.latex(f"S(x) = {c['a']:.4f} + {c['b']:.4f}(dx) + {c['c']:.4f}(dx)^2 + {c['d']:.4f}(dx)^3")
            
            tab1, tab2 = st.tabs(["1st Derivative Formula", "2nd Derivative Formula"])
            
            with tab1:
                st.markdown("**1. The General Formula:**")
                st.latex(r"S'(x) = b + 2c(x-x_i) + 3d(x-x_i)^2")
                st.markdown(f"**2. Substitution ($dx = {target_x} - {target_x-dx:.4f} = {dx:.4f}$):**")
                st.latex(f"{c['b']:.4f} + 2({c['c']:.4f})({dx:.4f}) + 3({c['d']:.4f})({dx:.4f})^2")
                st.success(f"**Result:** {res['dy']:.6f}")
            
            with tab2:
                st.markdown("**1. The General Formula:**")
                st.latex(r"S''(x) = 2c + 6d(x-x_i)")
                st.markdown(f"**2. Substitution ($dx = {dx:.4f}$):**")
                st.latex(f"2({c['c']:.4f}) + 6({c['d']:.4f})({dx:.4f})")
                st.success(f"**Result:** {res['d2y']:.6f}")

        # --- LEAST SQUARES ---
        elif "Least Squares" in method:
            st.markdown("### 📉 Least Squares (Parabolic Fit)")
            st.markdown("Fits a parabola $y = Ax^2 + Bx + C$ to the nearest 5 points to smooth out noise.")
            
            tab1, tab2 = st.tabs(["1st Derivative Formula", "2nd Derivative Formula"])
            
            with tab1:
                st.markdown("**1. The General Formula:**")
                st.latex(r"y' = 2Ax + B")
                st.markdown(f"**2. Substitution ($x={target_x}$):**")
                st.latex(f"2({A:.4f})({target_x}) + {B:.4f}")
                st.success(f"**Result:** {res['dy']:.6f}")
                
            with tab2:
                st.markdown("**1. The General Formula:**")
                st.latex(r"y'' = 2A")
                st.markdown(f"**2. Substitution:**")
                st.latex(f"2({A:.4f})")
                st.success(f"**Result:** {res['d2y']:.6f}")

        # --- FALLBACK ---
        else:
            # Unified handling for Central and Lagrange to match Forward/Backward UI
            tab1, tab2 = st.tabs(["1st Derivative Formula", "2nd Derivative Formula"])
            
            # Try to get symbolic expressions
            poly_str = res.get('expr')
            d1_latex = res.get('d1_expr', 'N/A')
            d2_latex = "N/A"
            
            if poly_str:
                try:
                    x_sym = sp.symbols('x')
                    # Clean string for sympy if needed
                    expr_sym = sp.sympify(str(poly_str).replace('^', '**'))
                    
                    d1_expr = sp.diff(expr_sym, x_sym, 1)
                    d2_expr = sp.diff(expr_sym, x_sym, 2)
                    
                    d1_latex = sp.latex(d1_expr)
                    d2_latex = sp.latex(d2_expr)
                except: pass
            
            with tab1:
                st.markdown(f"**1. The General Formula ({method}):**")
                st.markdown("For this method, we differentiate the interpolating polynomial $P(x)$:")
                st.latex(f"f'(x) \\approx P'(x) = {d1_latex}")
                st.markdown(f"**2. Substitution ($x={target_x}$):**")
                st.latex(f"f'({target_x}) = {res['dy']:.6f}")
                
            with tab2:
                st.markdown(f"**1. The General Formula ({method}):**")
                st.markdown("Differentiating $P(x)$ twice:")
                st.latex(f"f''(x) \\approx P''(x) = {d2_latex}")
                st.markdown(f"**2. Substitution ($x={target_x}$):**")
                st.latex(f"f''({target_x}) = {res['d2y']:.6f}")

    
        # --- 3. DIFFERENCE/COEFFICIENT TABLE ---
    if df_table is not None:
        with st.expander("Difference/Coefficient Table"):
            # 1. Create a copy so we don't break the original data for math later
            df_display = df_table.copy()
            
            # 2. Identify which columns to format
            exclude_cols = ["Interval", "Term", "i", "Step"]
            target_cols = [c for c in df_display.columns if c not in exclude_cols]

            # 3. Apply formatting directly to the data
            for col in target_cols:
                # Convert the column to String/Object type
                df_display[col] = df_display[col].apply(
                    lambda x: "" if pd.isna(x) or str(x).strip() == "None" 
                    else "{:.4f}".format(float(x))
                )

            # 4. Display the Clean Dataframe
            st.dataframe(df_display, use_container_width=True, height=250)