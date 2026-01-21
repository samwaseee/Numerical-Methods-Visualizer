import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_direct_steps(steps):
    st.subheader("Step-by-Step Elimination")
    
    for i, step in enumerate(steps):
        with st.expander(f"Step {i}: {step['description']}", expanded=(i==0 or i==len(steps)-1)):
            mat = np.array(step['matrix'])
            
            # Construct LaTeX array for augmented matrix
            rows_latex = []
            for row in mat:
                # Format numbers
                coeffs = " & ".join([f"{x:.4f}" for x in row[:-1]])
                const = f"{row[-1]:.4f}"
                rows_latex.append(f"{coeffs} & {const}")
            
            cols = len(mat[0])
            # Create format string like ccc|c
            col_fmt = "c" * (cols - 1) + "|c"
            
            latex_aug = f"\\left[\\begin{{array}}{{{col_fmt}}}\n"
            latex_aug += " \\\\ \n".join(rows_latex)
            latex_aug += "\n\\end{array}\\right]"
            
            st.latex(latex_aug)

def render_iterative_graph(df_history):
    st.subheader("Convergence Plot")
    
    if df_history.empty:
        st.warning("No iteration data generated (check for division by zero or divergence).")
        return

    fig = go.Figure()
    
    # Identify x columns (x1, x2, ...)
    x_cols = [c for c in df_history.columns if c.startswith('x')]
    
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
    
    st.markdown("### Iteration Table")
    st.dataframe(df_history, use_container_width=True)
