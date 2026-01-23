import streamlit as st
import numpy as np

def render_taylor_derivation(method):
    with st.expander("📘 Theory: Taylor Series Derivation", expanded=False):
        st.markdown(r"""
        **The Taylor Series Expansion** allows us to approximate the value of a function $y(x+h)$ based on its value and derivatives at $x$:
        
        $$y(x+h) = y(x) + h y'(x) + \frac{h^2}{2!} y''(x) + \frac{h^3}{3!} y'''(x) + \dots + \frac{h^n}{n!} y^{(n)}(x) + O(h^{n+1})$$
        """)
        
        if method == "Euler":
            st.markdown(r"""
            **Euler's Method** truncates the series after the first derivative (linear approximation):
            
            $$y(x+h) \approx y(x) + h y'(x)$$
            
            *   **Truncation Error:** The dominant term ignored is $\frac{h^2}{2} y''(x)$, so the local error is $O(h^2)$.
            *   **Global Error:** Accumulates to $O(h)$.
            """)
        elif method == "Heun":
            st.markdown(r"""
            **Heun's Method** (Improved Euler) approximates the second derivative term by averaging slopes:
            
            $$y(x+h) \approx y(x) + \frac{h}{2} [f(x, y) + f(x+h, y_{Euler})]$$
            
            This is equivalent to keeping terms up to $h^2$ in the Taylor series.
            *   **Local Error:** $O(h^3)$.
            *   **Global Error:** $O(h^2)$.
            """)
        elif method == "RK4":
            st.markdown(r"""
            **Runge-Kutta 4 (RK4)** uses a weighted average of 4 slopes ($k_1, k_2, k_3, k_4$) to match the Taylor Series up to the $h^4$ term:
            
            $$y(x+h) \approx y(x) + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
            
            It cancels out error terms up to the 4th derivative.
            *   **Local Error:** $O(h^5)$.
            *   **Global Error:** $O(h^4)$ (Very accurate).
            """)

def render_ode_steps(solver, method, df):
    render_taylor_derivation(method)
    st.markdown(f"### Step-by-Step: {method}")
    
    # Slider to select step
    n_steps = len(df) - 1
    if n_steps < 1:
        st.warning("Not enough data points to show steps.")
        return

    step_idx = st.slider("Select Step (Iteration)", 0, n_steps - 1, 0)
    
    row_curr = df.iloc[step_idx]
    row_next = df.iloc[step_idx + 1]
    
    x_i = float(row_curr['x'])
    y_i = float(row_curr['y'])
    h = float(row_next['x'] - x_i)
    
    f = solver.f
    
    with st.container(border=True):
        st.markdown(f"**Step {step_idx + 1}:** $x_{{{step_idx}}} = {x_i:.4f} \\to x_{{{step_idx+1}}} = {row_next['x']:.4f}$ ($h = {h:.4f}$)")
        st.markdown(f"**Current State:** $y_{{{step_idx}}} = {y_i:.6f}$")
        
        if method == "Euler":
            slope = f(x_i, y_i)
            st.markdown("#### Euler's Formula")
            st.latex(r"y_{i+1} = y_i + h \cdot f(x_i, y_i)")
            st.markdown("**Substitution:**")
            st.latex(f"f({x_i:.4f}, {y_i:.4f}) = {slope:.6f}")
            st.latex(f"y_{{{step_idx+1}}} = {y_i:.6f} + ({h:.4f})({slope:.6f})")
            st.latex(f"y_{{{step_idx+1}}} = {row_next['y']:.6f}")
            
        elif method == "Heun":
            # Recompute k values
            k1 = f(x_i, y_i)
            y_p = y_i + h * k1
            x_next_val = float(row_next['x'])
            k2 = f(x_next_val, y_p)
            
            st.markdown("#### Heun's Method (Predictor-Corrector)")
            st.latex(r"k_1 = f(x_i, y_i)")
            st.latex(r"y^p_{i+1} = y_i + h \cdot k_1 \quad \text{(Predictor)}")
            st.latex(r"k_2 = f(x_{i+1}, y^p_{i+1})")
            st.latex(r"y_{i+1} = y_i + \frac{h}{2}(k_1 + k_2) \quad \text{(Corrector)}")
            
            st.markdown("**Calculation:**")
            st.write(f"1. Slope at start: $k_1 = f({x_i:.4f}, {y_i:.4f}) = {k1:.6f}$")
            st.write(f"2. Predictor: $y^p = {y_i:.6f} + {h:.4f}({k1:.6f}) = {y_p:.6f}$")
            st.write(f"3. Slope at end: $k_2 = f({x_next_val:.4f}, {y_p:.6f}) = {k2:.6f}$")
            st.write(f"4. Average Slope: $(k_1 + k_2)/2 = {(k1+k2)/2:.6f}$")
            
            st.latex(f"y_{{{step_idx+1}}} = {y_i:.6f} + {h:.4f}({(k1+k2)/2:.6f}) = {row_next['y']:.6f}")

        elif method == "RK4":
            k1 = f(x_i, y_i)
            k2 = f(x_i + h/2, y_i + h*k1/2)
            k3 = f(x_i + h/2, y_i + h*k2/2)
            k4 = f(x_i + h, y_i + h*k3)
            
            st.markdown("#### Runge-Kutta 4th Order")
            c1, c2 = st.columns(2)
            with c1:
                st.latex(r"k_1 = f(x_i, y_i)")
                st.latex(r"k_2 = f(x_i + \frac{h}{2}, y_i + \frac{h}{2}k_1)")
            with c2:
                st.latex(r"k_3 = f(x_i + \frac{h}{2}, y_i + \frac{h}{2}k_2)")
                st.latex(r"k_4 = f(x_i + h, y_i + h k_3)")
            
            st.latex(r"y_{i+1} = y_i + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)")
            
            st.markdown("**Slopes:**")
            st.write(f"* $k_1 = {k1:.6f}$")
            st.write(f"* $k_2 = {k2:.6f}$")
            st.write(f"* $k_3 = {k3:.6f}$")
            st.write(f"* $k_4 = {k4:.6f}$")
            
            weighted_avg = (k1 + 2*k2 + 2*k3 + k4) / 6
            st.latex(f"y_{{{step_idx+1}}} = {y_i:.6f} + {h:.4f}({weighted_avg:.6f}) = {row_next['y']:.6f}")
            
        elif method == "RK45":
            st.info("RK45 is an adaptive step-size method. The step size $h$ changes based on error estimation between 4th and 5th order solutions.")
            st.write(f"Current step size used: $h = {h:.6f}$")
            st.write(f"Result: $y_{{{step_idx+1}}} = {row_next['y']:.6f}$")