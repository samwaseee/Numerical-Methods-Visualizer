import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_direct_steps(steps):
    st.subheader("Step-by-Step Elimination")
    
    for i, step in enumerate(steps):
        with st.expander(f"Step {i}: {step['description']}", expanded=(i==0 or i==len(steps)-1)):
            if "explanation" in step:
                st.info(step["explanation"])

            mat = np.array(step['matrix'])
            
            # Construct LaTeX array for augmented matrix
            rows_latex = []
            for row in mat:
                # Format numbers
                coeffs = " & ".join([f"{x:.4f}" for x in row[:-1]])
                const = f"{row[-1]:.4f}"
                rows_latex.append(f"{coeffs} & {const}")
            
            rows = len(mat)
            cols = len(mat[0])
            
            # Determine format string based on dimensions
            if cols == rows + 1:
                col_fmt = "c" * rows + "|c"  # Standard [A|B]
            elif cols == 2 * rows:
                col_fmt = "c" * rows + "|" + "c" * rows # Inversion [A|I]
            else:
                col_fmt = "c" * cols # Fallback
            
            latex_aug = f"\\left[\\begin{{array}}{{{col_fmt}}}\n"
            latex_aug += " \\\\ \n".join(rows_latex)
            latex_aug += "\n\\end{array}\\right]"
            
            st.latex(latex_aug)

def render_iterative_graph(df_history, spectral_radius=None):
    st.subheader("Convergence Plot")
    
    if df_history is None or df_history.empty:
        st.warning("No iteration data generated.")
        return

    if not df_history.empty and "Error" in df_history.columns:
        first_val = df_history.iloc[0]["Error"]
        if isinstance(first_val, str):
            st.error(first_val)
            st.dataframe(df_history, use_container_width=True)
            return

    # Display Spectral Radius Metric
    if spectral_radius is not None:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Spectral Radius (ρ)", f"{spectral_radius:.4f}", 
                help="Convergence is guaranteed if ρ < 1. Smaller is faster.")
        with c2:
            if spectral_radius >= 1:
                st.warning("ρ ≥ 1: Method may NOT converge.", icon="⚠️")
            else:
                st.success("ρ < 1: Convergence guaranteed.", icon="✅")

    # Render Graph
    try:
        fig = go.Figure()
        
        # Identify x columns (x1, x2, ...)
        x_cols = [c for c in df_history.columns if c.startswith('x')]
        
        if x_cols:
            for col in x_cols:
                fig.add_trace(go.Scatter(
                    x=df_history['Iter'],
                    y=df_history[col],
                    mode='lines+markers',
                    name=col
                ))
            
            fig.update_layout(
                title="Value of Unknowns per Iteration",
                xaxis_title="Iteration",
                yaxis_title="Value",
                height=450,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No variable data found to plot.")
            
    except Exception as e:
        st.error(f"Error rendering graph: {e}")
    
    st.markdown("### Iteration Table")
    st.dataframe(df_history, use_container_width=True)

def render_system_analysis(analysis):
    st.subheader("System Analysis (The 'Doctor')")
    
    # 1. Solvability
    c1, c2, c3 = st.columns(3)
    with c1:
        if analysis['determinant'] is not None:
            st.metric("Determinant", f"{analysis['determinant']:.4f}")
        else:
            st.metric("Determinant", "N/A")
    with c2:
        st.metric("Rank(A)", f"{analysis['rank_A']}")
    with c3:
        st.metric("Rank(A|B)", f"{analysis['rank_Aug']}")
    
    status = analysis['solvability']
    if "Unique" in status:
        st.success(f"**Status:** {status}", icon=":material/check_circle:")
    elif "Infinite" in status:
        st.warning(f"**Status:** {status}", icon=":material/warning:")
    else:
        st.error(f"**Status:** {status}", icon=":material/error:")
    st.caption(analysis['solvability_desc'])
    
    # 2. Condition Number
    cond = analysis['condition_number']
    if cond is not None:
        # Gauge Chart Logic
        max_val = 1500
        display_val = cond
        
        if cond == float('inf'):
            display_val = max_val
            bar_color = "#ef4444" # Red
            suffix = " (Inf)"
        else:
            suffix = ""
            if cond > 1000:
                bar_color = "#ef4444" # Red
                max_val = max(1500, cond * 1.1)
            elif cond > 100:
                bar_color = "#f59e0b" # Amber
            else:
                bar_color = "#10b981" # Emerald

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = display_val,
            number = {'suffix': suffix, 'font': {'size': 24}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Condition Number (Stability)", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [1, max_val], 'tickwidth': 1},
                'bar': {'color': bar_color},
                'steps': [
                    {'range': [1, 100], 'color': "rgba(16, 185, 129, 0.15)"},   # Green zone
                    {'range': [100, 1000], 'color': "rgba(245, 158, 11, 0.15)"}, # Orange zone
                    {'range': [1000, max_val], 'color': "rgba(239, 68, 68, 0.15)"} # Red zone
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1000
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=30, r=30, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("The **Condition Number** measures how sensitive the solution is to changes in the input data. A high value means the matrix is 'ill-conditioned' and prone to large numerical errors.")

        if cond > 1000:
            st.error(f"**Ill-conditioned!** Small input errors will be magnified by {cond:.1f}x.", icon=":material/dangerous:")
        elif cond > 100:
            st.warning(f"**Poorly conditioned.** Precision loss is likely.", icon=":material/warning:")
        else:
            st.success(f"**Stable System.** Solution is reliable.", icon=":material/thumb_up:")

    # 3. Diagonal Dominance
    with st.expander("Diagonal Dominance Check (For Iterative Methods)"):
        dom_data = analysis['diagonal_dominance']
        if dom_data['is_dominant']:
            st.success("Matrix is Diagonally Dominant. Iterative methods are guaranteed to converge.")
        else:
            st.warning("Matrix is NOT Diagonally Dominant. Iterative methods might diverge.")
        
        # Table of rows
        rows_data = []
        for r in dom_data['rows']:
            sign = ">=" if r['is_dominant'] else "<"
            status = "✅" if r['is_dominant'] else "❌"
            rows_data.append({
                "Row": f"Row {r['row']}",
                "Diagonal |aii|": f"{r['diag']:.4f}",
                "Check": sign,
                "Sum Off-Diag": f"{r['sum_off']:.4f}",
                "Status": status
            })
        st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)

        reorder_info = analysis.get('reorder_for_dominance')
        if reorder_info and reorder_info['is_suggested']:
            st.info(f"**Reordering Suggestion:** {reorder_info['status_message']}")
        elif reorder_info:
            st.caption(f"**Reordering Attempt:** {reorder_info['status_message']}")
    
    # 4. Eigenvalues & Eigenvectors (New)
    with st.expander("Eigenvalues & Eigenvectors (Spectrum)"):
        if analysis.get('eigen'):
            st.caption("The eigenvalues ($\lambda$) and corresponding eigenvectors ($v$) satisfy $Av = \lambda v$.")
            
            # Create two columns: One for Values, one for Vectors
            c_eig1, c_eig2 = st.columns([1, 2])
            
            with c_eig1:
                st.markdown("**Eigenvalues ($\lambda$)**")
                for e in analysis['eigen']:
                    val = e['value']
                    # Format complex numbers nicely
                    if np.iscomplex(val):
                        fmt_val = f"{val.real:.4f} + {val.imag:.4f}j"
                    else:
                        fmt_val = f"{val.real:.4f}"
                    st.code(f"λ{e['index']} = {fmt_val}")
            
            with c_eig2:
                st.markdown("**Eigenvectors ($v$)**")
                for e in analysis['eigen']:
                    vec = np.array(e['vector'])
                    # Format vector nicely
                    if np.iscomplexobj(vec):
                        fmt_vec = [f"{x.real:.2f}+{x.imag:.2f}j" for x in vec]
                    else:
                        fmt_vec = [f"{x.real:.4f}" for x in vec]
                    st.code(f"v{e['index']} = {fmt_vec}")
        else:
            st.info("Eigenvalues are only calculated for square matrices.")

    st.divider()
