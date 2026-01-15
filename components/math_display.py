import streamlit as st
import pandas as pd

def display_math_analysis(method, row, solver, extra_params=None):
    st.subheader("🧮 Step-by-Step Analysis")
    
    # --- 1. SCANNING ANALYSIS ---
    if method == "Find All Roots (Scan)":
        # FIX: Updated keys to match new algorithms.py format
        x_start = row['Interval Start']
        x_end = row['Interval End']
        f_start = row['f(Start)']
        f_end = row['f(End)']
        sign_change = row['Sign Change?']
        
        st.markdown(f"""
        **Current Interval:** $[{x_start:.4f}, {x_end:.4f}]$
        
        1. **Evaluate Function at Boundaries:**
           $$f({x_start:.4f}) = {f_start:.4f}$$
           $$f({x_end:.4f}) = {f_end:.4f}$$
        
        2. **Check for Sign Change (Intermediate Value Theorem):**
           Multiply the values:
           $$ {f_start:.4f} \\times {f_end:.4f} = {f_start * f_end:.4f} $$
        """)
        
        if "YES" in str(sign_change):
            st.success(f"""
            **Result:** Negative Product (< 0).
            
            ✅ **Sign Change Detected!**
            According to the Intermediate Value Theorem, since $f(x)$ is continuous and changes sign, 
            there must be at least one root in this interval.
            """)
        else:
            st.info(f"""
            **Result:** Positive Product (> 0).
            
            ❌ **No Sign Change.**
            The function does not cross the x-axis in this specific step. Moving to next interval...
            """)

    # --- 2. BISECTION / FALSE POSITION ---
    elif method in ["Bisection Method", "False Position"]:
        a = row['a']
        b = row['b']
        c = row['c (root)']
        fc = row['f(c)']
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        
        if method == "Bisection Method":
            st.latex(r"c = \frac{a + b}{2} = \frac{" + f"{a:.4f} + {b:.4f}" + r"}{2} = " + f"{c:.4f}")
        else:
            fa = solver.f(a)
            fb = solver.f(b)
            st.latex(r"c = \frac{a f(b) - b f(a)}{f(b) - f(a)}")
            st.caption(f"Substituting values: a={a:.4f}, b={b:.4f}, f(a)={fa:.4f}, f(b)={fb:.4f}")
            st.markdown(f"**Result:** $c = {c:.4f}$")

        st.markdown(f"**Check Sign:** $f(c) = {fc:.6f}$")
        
        if solver.f(a) * fc < 0:
            st.markdown(f"Since $f(a) \\cdot f(c) < 0$, the root lies in left sub-interval **[{a:.4f}, {c:.4f}]**.")
        else:
            st.markdown(f"Since $f(a) \\cdot f(c) > 0$, the root lies in right sub-interval **[{c:.4f}, {b:.4f}]**.")

    # --- 3. NEWTON-RAPHSON ---
    elif method == "Newton-Raphson":
        x_old = row['x_old']
        fx = row['f(x)']
        dfx = row["f'(x)"]
        x_new = row['x_new']
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        st.latex(r"x_{new} = x_{old} - \frac{f(x_{old})}{f'(x_{old})}")
        
        st.markdown(f"""
        1. **Evaluate Point:** $x_{{old}} = {x_old:.6f}$
           $$f({x_old:.4f}) = {fx:.6f}$$
           $$f'({x_old:.4f}) = {dfx:.6f}$$
           
        2. **Apply Formula:**
           $$x_{{new}} = {x_old:.6f} - \\frac{{{fx:.6f}}}{{{dfx:.6f}}}$$
           $$x_{{new}} = {x_new:.6f}$$
        """)

    # --- 4. SECANT METHOD ---
    elif method == "Secant Method":
        x_prev = row['x_prev']
        x_curr = row['x_curr']
        x_new = row['x_new']
        fx_prev = solver.f(x_prev)
        fx_curr = solver.f(x_curr)
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        st.latex(r"x_{new} = x_{curr} - f(x_{curr}) \frac{x_{curr} - x_{prev}}{f(x_{curr}) - f(x_{prev})}")
        
        st.markdown(f"""
        Using points: $x_{{i-1}} = {x_prev:.4f}$ and $x_i = {x_curr:.4f}$
        
        $$x_{{new}} = {x_curr:.4f} - ({fx_curr:.4f}) \\frac{{{x_curr:.4f} - {x_prev:.4f}}}{{{fx_curr:.4f} - {fx_prev:.4f}}}$$
        
        **New Estimate:** $x_{{new}} = {x_new:.6f}$
        """)

    # --- 5. FIXED POINT ---
    elif method == "Fixed Point Iteration":
        x_old = row['x_old']
        x_new = row['x_new']
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        st.latex(r"x_{new} = g(x_{old})")
        
        st.markdown(f"""
        Substitute $x_{{old}} = {x_old:.6f}$ into your rearranged function $g(x)$:
        
        $$x_{{new}} = g({x_old:.6f}) = {x_new:.6f}$$
        
        **Error Check:**
        $$E_a = \\left| \\frac{{{x_new:.6f} - {x_old:.6f}}}{{{x_new:.6f}}} \\right| \\times 100 = {row.get('Error % (Ea)', 0):.4f}\\%$$
        """)