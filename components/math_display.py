import streamlit as st
import pandas as pd

def display_math_analysis(method, row, solver, extra_params=None):
    st.subheader("🧮 Step-by-Step Analysis")
    
    # --- 1. SCANNING ANALYSIS ---
    if method == "Find All Roots (Scan)":
        x_start = row['Interval Start']
        x_end = row['Interval End']
        f_start = row['f(Start)']
        f_end = row['f(End)']
        sign_change = row['Sign Change?']
        
        st.markdown(f"""
        **Current Interval:** $[{x_start}, {x_end}]$
        
        1. **Evaluate Function at Boundaries:**
           $$f({x_start}) = {f_start}$$
           $$f({x_end}) = {f_end}$$
        
        2. **Check for Sign Change (Intermediate Value Theorem):**
           Multiply the values:
           $$ {f_start} \\times {f_end} = {f_start * f_end} $$
        """)
        
        if "YES" in str(sign_change):
            st.success(f"""
            **Result:** Negative Product (< 0).
            ✅ **Sign Change Detected!**
            """)
        else:
            st.info(f"""
            **Result:** Positive Product (> 0).
            ❌ **No Sign Change.**
            """)

    # --- 2. BISECTION / FALSE POSITION ---
    elif method in ["Bisection Method", "False Position"]:
        a = row['a']
        b = row['b']
        c = row['c (root)']
        fc = row['f(c)']
        
        # Theory Tab
        with st.expander("📚 How It Works", expanded=False):
            if method == "Bisection Method":
                st.markdown("""
                **Bisection Method** divides the interval in half at each step.
                
                - **Principle:** Intermediate Value Theorem - if f(a)·f(b) < 0, a root exists in [a,b]
                - **Strategy:** Always cut the interval in half
                - **Pros:** Guaranteed convergence, stable, simple
                - **Cons:** Slow (linear convergence), doesn't use derivatives
                - **Best for:** When you have a guaranteed bracket and want reliability
                """)
            else:
                st.markdown("""
                **False Position Method** uses weighted division based on function values.
                
                - **Principle:** Linear interpolation between two points
                - **Strategy:** Cut proportionally based on f(a) and f(b) magnitudes
                - **Pros:** Faster than bisection when one endpoint is closer to root
                - **Cons:** Can converge slowly if root is off-center
                - **Best for:** When function behavior is smooth between endpoints
                """)
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        
        if method == "Bisection Method":
            st.latex(r"c = \frac{a + b}{2} = \frac{" + f"{a} + {b}" + r"}{2} = " + f"{c}")
        else:
            fa = solver.f(a)
            fb = solver.f(b)
            st.latex(r"c = \frac{a f(b) - b f(a)}{f(b) - f(a)}")
            st.caption(f"Substituting values: a={a}, b={b}, f(a)={fa}, f(b)={fb}")
            st.markdown(f"**Result:** $c = {c}$")

        st.markdown(f"**Check Sign:** $f(c) = {fc}$")
        
        # Convergence Analysis
        fa = solver.f(a)
        interval_width = b - a
        converging = abs(fc) < abs(fa) and abs(fc) < abs(solver.f(b))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Interval Width", f"{interval_width}")
        with col2:
            st.metric("f(c) Value", f"{fc}")
        with col3:
            status = "✅ Converging" if converging else "⏳ Still narrowing"
            st.metric("Status", status)
        
        if solver.f(a) * fc < 0:
            st.markdown(f"Since $f(a) \\cdot f(c) < 0$, the root lies in left sub-interval **[{a}, {c}]**.")
        else:
            st.markdown(f"Since $f(a) \\cdot f(c) > 0$, the root lies in right sub-interval **[{c}, {b}]**.")

    # --- 3. NEWTON-RAPHSON ---
    elif method == "Newton-Raphson":
        x_old = row['x_old']
        fx = row['f(x)']
        dfx = row["f'(x)"]
        x_new = row['x_new']
        
        # Theory Tab
        with st.expander("📚 How It Works", expanded=False):
            st.markdown("""
            **Newton-Raphson Method** uses tangent lines to find roots.
            
            - **Principle:** Follow the tangent line to where it crosses the x-axis
            - **Formula:** $x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}$
            - **Convergence:** Quadratic (very fast near root)
            - **Pros:** Fastest convergence, needs only f and f'
            - **Cons:** Fails if f'(x)=0, needs good initial guess, no bracket needed
            - **Best for:** When you have derivative and good starting point
            """)
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        st.latex(r"x_{new} = x_{old} - \frac{f(x_{old})}{f'(x_{old})}")
        
        st.markdown(f"""
        1. **Evaluate Point:** $x_{{old}} = {x_old}$
           $$f({x_old}) = {fx}$$
           $$f'({x_old}) = {dfx}$$
           
        2. **Apply Formula:**
           $$x_{{new}} = {x_old} - \\frac{{{fx}}}{{{dfx}}}$$
           $$x_{{new}} = {x_new}$$
        """)
        
        # Convergence Analysis
        step_size = abs(x_new - x_old)
        error_reduction = abs(fx) if row['Iter'] == 1 else abs(fx)
        converging = step_size < 0.001
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Step Size", f"{step_size}")
        with col2:
            st.metric("|f(x)|", f"{abs(fx)}")
        with col3:
            status = "✅ Converged" if converging else "🚀 Fast approach"
            st.metric("Status", status)

    # --- 4. SECANT METHOD ---
    elif method == "Secant Method":
        x_prev = row['x_prev']
        x_curr = row['x_curr']
        x_new = row['x_new']
        fx_prev = solver.f(x_prev)
        fx_curr = solver.f(x_curr)
        
        # Theory Tab
        with st.expander("📚 How It Works", expanded=False):
            st.markdown("""
            **Secant Method** approximates the derivative using two points.
            
            - **Principle:** Draw a line (secant) through two recent points
            - **Formula:** Slope ≈ $\\frac{f(x_i) - f(x_{i-1})}{x_i - x_{i-1}}$ (no derivative needed!)
            - **Convergence:** Superlinear (faster than bisection, slower than Newton)
            - **Pros:** No derivative needed, only needs two starting points
            - **Cons:** Less stable than Newton, convergence not guaranteed
            - **Best for:** When you can't compute derivative
            """)
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        st.latex(r"x_{new} = x_{curr} - f(x_{curr}) \frac{x_{curr} - x_{prev}}{f(x_{curr}) - f(x_{prev})}")
        
        st.markdown(f"""
        Using points: $x_{{i-1}} = {x_prev}$ and $x_i = {x_curr}$
        
        **Approximate Slope:** $m \\approx \\frac{{f(x_i) - f(x_{{i-1}})}}{{x_i - x_{{i-1}}}} = \\frac{{{fx_curr} - {fx_prev}}}{{{x_curr} - {x_prev}}} = {(fx_curr - fx_prev)/(x_curr - x_prev)}$
        
        $$x_{{new}} = {x_curr} - ({fx_curr}) \\frac{{{x_curr} - {x_prev}}}{{{fx_curr} - {fx_prev}}}$$
        
        **New Estimate:** $x_{{new}} = {x_new}$
        """)
        
        # Convergence Analysis
        step_size = abs(x_new - x_curr)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Step Size", f"{step_size}")
        with col2:
            st.metric("|f(x)|", f"{abs(fx_curr)}")
        with col3:
            status = "✅ Good" if step_size < 0.01 else "⏳ Progressing"
            st.metric("Status", status)

    # --- 5. FIXED POINT ---
    elif method == "Fixed Point Iteration":
        x_old = row['x_old']
        x_new = row['x_new']
        
        # Theory Tab
        with st.expander("📚 How It Works", expanded=False):
            st.markdown("""
            **Fixed Point Iteration** rearranges f(x)=0 to x=g(x) form.
            
            - **Principle:** Find x where x = g(x)
            - **Rearrangement:** From f(x)=0, derive x = g(x) (infinite forms possible)
            - **Convergence:** Linear (if |g'(x)| < 1 near root)
            - **Pros:** Simple concept, flexible formula
            - **Cons:** Heavily depends on g(x) choice, can diverge
            - **Key Rule:** Need $|g'(x)| < 1$ at the root for convergence!
            - **Best for:** When you can rearrange to find a good g(x)
            """)
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        st.latex(r"x_{new} = g(x_{old})")
        
        try:
            g_prime = (solver.f(x_old + 0.0001) - solver.f(x_old)) / 0.0001
            convergence_check = abs(g_prime) < 1
            status_text = "✅ Will converge" if convergence_check else "❌ May diverge"
        except:
            status_text = "⚠️ Check convergence"
            g_prime = None
        
        st.markdown(f"""
        Substitute $x_{{old}} = {x_old}$ into your rearranged function $g(x)$:
        $$x_{{new}} = g({x_old}) = {x_new}$$
        """)
        
        # Convergence Analysis
        step_size = abs(x_new - x_old)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Step Size", f"{step_size}")
        with col2:
            if g_prime is not None:
                st.metric("|g'(x)| est", f"{abs(g_prime)}")
            else:
                st.metric("Status", status_text)
        with col3:
            st.metric("Convergence", status_text)

    # --- 6. HALLEY'S METHOD ---
    elif method == "Halley's Method":
        x_old = row['x_old']
        fx = row['f(x)']
        dfx = row["f'(x)"]
        d2fx = row["f''(x)"]
        x_new = row['x_new']
        
        # Theory Tab
        with st.expander("📚 How It Works", expanded=False):
            st.markdown("""
            **Halley's Method** uses first AND second derivatives for faster convergence.
            
            - **Principle:** Fit a better curve using curvature information
            - **Convergence:** Cubic (faster than Newton's quadratic!)
            - **Formula:** Uses f, f', f'' to find root
            - **Pros:** Fastest convergence rate, excellent near root
            - **Cons:** Needs second derivative, more complex
            - **Compare to Newton:** About 3x faster per iteration
            - **Best for:** When you need very fast convergence and have f''
            """)
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        st.latex(r"x_{new} = x_n - \frac{2 f(x_n) f'(x_n)}{2 [f'(x_n)]^2 - f(x_n) f''(x_n)}")
        
        st.markdown(f"""
        **Step-by-Step:**
        * $f(x) = {fx}$
        * $f'(x) = {dfx}$ (first derivative - slope)
        * $f''(x) = {d2fx}$ (second derivative - curvature)
        
        **Numerator:** $2 f(x) f'(x) = 2 \\times {fx} \\times {dfx} = {2*fx*dfx}$
        
        **Denominator:** $2[f'(x)]^2 - f(x)f''(x) = 2({dfx})^2 - ({fx})({d2fx}) = {2*dfx**2 - fx*d2fx}$
        
        $$x_{{new}} = {x_new}$$
        """)
        
        # Convergence Analysis
        step_size = abs(x_new - x_old)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Step Size", f"{step_size}")
        with col2:
            st.metric("|f(x)|", f"{abs(fx)}")
        with col3:
            status = "🚀 Very fast" if step_size < 0.0001 else "✅ Converging fast"
            st.metric("Status", status)

    # --- 7. MULLER'S METHOD ---
    elif method == "Muller's Method":
        p0 = row['p0']
        p2 = row['x_old']
        b_coef = row.get('b_coef', 0)
        D = row.get('D', 0)
        
        # Theory Tab
        with st.expander("📚 How It Works", expanded=False):
            st.markdown("""
            **Muller's Method** fits a parabola through 3 points.
            
            - **Principle:** Approximate function with parabola, find where it crosses x-axis
            - **Formula:** Use quadratic interpolation through 3 points
            - **Convergence:** Superlinear (order ≈ 1.84)
            - **Pros:** Can find complex roots, no derivative needed, 3 starting points
            - **Cons:** More complex math, needs 3 points, sometimes unstable
            - **Unique Feature:** Can find COMPLEX ROOTS!
            - **Best for:** When you want to find complex roots or have no derivatives
            """)
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        st.markdown("🔷 Fitting a parabola through 3 recent points...")
        
        st.latex(r"x_{new} = p_2 - \frac{2c}{b \pm \sqrt{b^2 - 4ac}}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Previous Point", f"{p0}")
        with col2:
            st.metric("Coefficient b", f"{b_coef}")
        with col3:
            if isinstance(D, complex):
                st.warning("Complex discriminant!")
            else:
                st.metric("Discriminant", f"{D}")
        
        if isinstance(D, complex):
            st.warning(f"⚠️ Complex Root Found: Discriminant = {D}\n\nThis method can find complex roots! Result uses complex arithmetic.")
        
        st.markdown(f"""
        **Previous Points:** $p_0 = {p0}, p_2 = {p2}$
        **Quadratic Coefficients:** $b = {b_coef}$
        
        **New Root Estimate:**
        $$x_{{new}} = {row['x_new']}$$
        """)

    # --- 8. BRENT'S METHOD ---
    elif method == "Brent's Method":
        strategy = row.get('Method Used', 'Unknown')
        x_new = row['x_new']
        x_old = row.get('x_old', 0)
        c_root = row.get('c (root)', x_new)
        f_x = row.get('f(x)', 0)
        
        # Theory Tab
        with st.expander("📚 How It Works", expanded=False):
            st.markdown("""
            **Brent's Method** is an ADAPTIVE hybrid algorithm.
            
            - **Principle:** Try fast interpolation, fall back to bisection if needed
            - **Hybrid Strategy:** Starts with inverse quadratic interpolation
            - **Fallback:** Switches to bisection if interpolation fails
            - **Convergence:** Superlinear (when interpolation works)
            - **Pros:** **Fastest** for most problems, guaranteed convergence, adaptive
            - **Cons:** More complex implementation
            - **Why Hybrid?:** Combines speed of interpolation with reliability of bisection
            - **Best for:** General root finding - often the best choice!
            """)
        
        st.markdown(f"**Iteration {row['Iter']}:**")
        
        # Strategy indicator with colors and formulas
        if strategy == "Bisection":
            st.warning(f"⚡ **Strategy:** {strategy} (safe fallback)")
            st.latex(r"x_{new} = \frac{a + b}{2}")
            st.markdown("""
            **Why fallback?** The interpolation estimate was outside [a,b] or too unreliable.
            Bisection is slower but guarantees the bracket shrinks.
            """)
        elif strategy == "Converged":
            st.success(f"✅ **Status:** {strategy}")
            st.markdown(f"Root found! $x_{{new}} = {x_new}$")
        elif strategy == "No Bracket":
            st.error(f"❌ **Problem:** {strategy}")
            st.markdown("f(a) and f(b) have same sign - cannot guarantee a root in [a,b]")
        else:
            st.info(f"⚡ **Strategy:** {strategy} (quadratic interpolation)")
            st.latex(r"x_{new} = x_{old} - \frac{f(x_{old})}{\text{Inverse Quadratic Approximation}}")
            st.markdown("""
            **Inverse Quadratic Interpolation (IQI):**
            Uses 3 points to fit a parabola, then finds where it crosses x-axis.
            
            This is faster than bisection because it uses the function's behavior,
            not just bracket endpoints.
            """)
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("New Estimate", f"{x_new}")
        with col2:
            try:
                f_val = solver.f(x_new)
                st.metric("|f(x)|", f"{abs(f_val)}")
            except:
                st.metric("Status", "Evaluated")
        with col3:
            st.metric("Strategy", strategy[:10])