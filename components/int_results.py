import streamlit as st

def render_general_quadrature_formula(method=None):
    with st.container(border=True):
        st.markdown("### 📘 General Quadrature Formula")
        st.markdown("The General Quadrature Formula is derived by integrating Newton's Forward Difference formula. All specific integration methods are special cases of this formula.")
        
        st.latex(r"""
            I = h \left[ n y_0 + \frac{n^2}{2} \Delta y_0 
            + \frac{1}{2} \left( \frac{n^3}{3} - \frac{n^2}{2} \right) \Delta^2 y_0 
            + \frac{1}{6} \left( \frac{n^4}{4} - n^3 + n^2 \right) \Delta^3 y_0 
            + \dots \right]
        """)
        
        st.markdown("**Derivations:**")
        derivations = {
            "Trapezoidal Rule": "* **Trapezoidal Rule:** Set $n=1$ (Linear approximation)",
            "Simpson's 1/3 Rule": "* **Simpson's 1/3 Rule:** Set $n=2$ (Parabolic approximation)",
            "Simpson's 3/8 Rule": "* **Simpson's 3/8 Rule:** Set $n=3$ (Cubic approximation)",
            "Boole's Rule": "* **Boole's Rule:** Set $n=4$ (Quartic approximation)",
            "Weddle's Rule": "* **Weddle's Rule:** Set $n=6$ (Higher order approximation)",
        }
        
        if method and method in derivations:
            st.markdown(derivations[method])
        else:
            st.markdown("\n".join(derivations.values()))

def render_integration_results(res, details, method):
    """
    Dispatcher for rendering integration results.
    """
    st.markdown(f"### Result: {res:.6f}")
    
    with st.expander("Show Step-by-Step Calculation", expanded=True, icon=":material/calculate:"):
        if method == "Trapezoidal Rule":
            _render_trapezoidal_steps(res, details)
        elif method == "Simpson's 1/3 Rule":
            _render_simpson13_steps(res, details)
        elif method == "Simpson's 3/8 Rule":
            _render_simpson38_steps(res, details)
        elif method == "Boole's Rule":
            _render_booles_steps(res, details)
        elif method == "Weddle's Rule":
            _render_weddles_steps(res, details)

def _render_trapezoidal_steps(res, d):
    st.markdown("#### Trapezoidal Rule Formula")
    st.latex(r"I \approx \frac{h}{2} \left[ (y_0 + y_n) + 2 \sum_{i=1}^{n-1} y_i \right]")
    
    st.markdown("#### Substitution")
    st.markdown(f"""
    * **Step size ($h$):** {d['h']:.4f}
    * **Ends ($y_0 + y_n$):** {d['y0']:.4f} + {d['yn']:.4f} = {d['y0'] + d['yn']:.4f}
    * **Middle Sum ($\sum y_i$):** {d['sum_mid']:.4f}
    """)
    
    st.latex(f"I \\approx \\frac{{{d['h']:.4f}}}{{2}} \\left[ ({d['y0'] + d['yn']:.4f}) + 2({d['sum_mid']:.4f}) \\right]")
    st.latex(f"I \\approx {res:.6f}")

def _render_simpson13_steps(res, d):
    st.markdown("#### Simpson's 1/3 Rule Formula")
    st.latex(r"I \approx \frac{h}{3} \left[ (y_0 + y_n) + 4 \sum y_{odd} + 2 \sum y_{even} \right]")
    
    st.markdown("#### Substitution")
    st.markdown(f"""
    * **Step size ($h$):** {d['h']:.4f}
    * **Ends ($y_0 + y_n$):** {d['y0'] + d['yn']:.4f}
    * **Odd Sum ($\sum y_{{1,3...}}$):** {d['sum_odd']:.4f}
    * **Even Sum ($\sum y_{{2,4...}}$):** {d['sum_even']:.4f}
    """)
    
    st.latex(f"I \\approx \\frac{{{d['h']:.4f}}}{{3}} \\left[ ({d['y0'] + d['yn']:.4f}) + 4({d['sum_odd']:.4f}) + 2({d['sum_even']:.4f}) \\right]")
    st.latex(f"I \\approx {res:.6f}")

def _render_simpson38_steps(res, d):
    st.markdown("#### Simpson's 3/8 Rule Formula")
    st.latex(r"I \approx \frac{3h}{8} \left[ (y_0 + y_n) + 3 \sum y_{others} + 2 \sum y_{multiple\_of\_3} \right]")
    
    st.markdown("#### Substitution")
    st.markdown(f"""
    * **Step size ($h$):** {d['h']:.4f}
    * **Ends ($y_0 + y_n$):** {d['y0'] + d['yn']:.4f}
    * **Others Sum ($\sum y_{{1,2,4,5...}}$):** {d['sum_others']:.4f}
    * **Multiples of 3 Sum ($\sum y_{{3,6...}}$):** {d['sum_mult3']:.4f}
    """)
    
    st.latex(f"I \\approx \\frac{{3({d['h']:.4f})}}{{8}} \\left[ ({d['y0'] + d['yn']:.4f}) + 3({d['sum_others']:.4f}) + 2({d['sum_mult3']:.4f}) \\right]")
    st.latex(f"I \\approx {res:.6f}")

def _render_booles_steps(res, d):
    st.markdown("#### Boole's Rule Formula")
    st.latex(r"I \approx \frac{2h}{45} \left[ 7(y_0+y_n) + 32\sum y_{1,3,5..} + 12\sum y_{2,6,10..} + 14\sum y_{4,8,..} \right]")
    
    st.markdown("#### Substitution")
    st.markdown(f"""
    * **Step size ($h$):** {d['h']:.4f}
    * **Ends ($y_0 + y_n$):** {d['y0'] + d['yn']:.4f}
    * **Sum with coeff 32 ($y_1, y_3, ...$):** {d['sum_32']:.4f}
    * **Sum with coeff 12 ($y_2, y_6, ...$):** {d['sum_12']:.4f}
    * **Sum with coeff 14 (junctions $y_4, y_8, ...$):** {d['sum_14']:.4f}
    """)
    
    st.latex(f"I \\approx \\frac{{2({d['h']:.4f})}}{{45}} \\left[ 7({d['y0'] + d['yn']:.4f}) + 32({d['sum_32']:.4f}) + 12({d['sum_12']:.4f}) + 14({d['sum_14']:.4f}) \\right]")
    st.latex(f"I \\approx {res:.6f}")

def _render_weddles_steps(res, d):
    st.markdown("#### Weddle's Rule Formula")
    st.latex(r"I \approx \frac{3h}{10} \left[ (y_0+y_n) + 5\sum y_{1,5,7..} + \sum y_{2,4,8..} + 6\sum y_{3,9,15..} + 2\sum y_{6,12,..} \right]")
    
    st.markdown("#### Substitution")
    st.markdown(f"""
    * **Step size ($h$):** {d['h']:.4f}
    * **Ends ($y_0 + y_n$):** {d['y0'] + d['yn']:.4f}
    * **Sum with coeff 5 ($y_1, y_5, ...$):** {d['sum_5']:.4f}
    * **Sum with coeff 1 ($y_2, y_4, ...$):** {d['sum_1']:.4f}
    * **Sum with coeff 6 ($y_3, y_9, ...$):** {d['sum_6']:.4f}
    * **Sum with coeff 2 (junctions $y_6, y_{12}, ...$):** {d['sum_2']:.4f}
    """)
    
    st.latex(f"I \\approx \\frac{{3({d['h']:.4f})}}{{10}} \\left[ ({d['y0'] + d['yn']:.4f}) + 5({d['sum_5']:.4f}) + ({d['sum_1']:.4f}) + 6({d['sum_6']:.4f}) + 2({d['sum_2']:.4f}) \\right]")
    st.latex(f"I \\approx {res:.6f}")
