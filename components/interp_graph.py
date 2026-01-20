import plotly.graph_objects as go
import streamlit as st

def render_interpolation_chart(solver, original_df, eval_pt=None, inverse=False, show_tangents=False):
    st.subheader("Visualizing the Curve")
    
    # 1. Get Curve Points
    # solver.get_eval_points() returns (input_axis_vals, output_axis_vals)
    range_vals, computed_vals = solver.get_eval_points()
    
    if not len(range_vals):
        st.warning("Not enough data to plot.")
        return

    # 2. Handle Inverse Axis Swapping
    if inverse:
        # Solver Input was Y, Output was X.
        # We want to plot X on horizontal, Y on vertical.
        x_curve = computed_vals
        y_curve = range_vals
    else:
        x_curve = range_vals
        y_curve = computed_vals
        
    fig = go.Figure()
    
    # 3. Plot The Polynomial Curve
    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve, 
        mode='lines', name='Polynomial',
        line=dict(color='#6366f1', width=3)
    ))
    
    # --- Plot Tangents (Hermite) ---
    if show_tangents and 'yp' in original_df.columns and not inverse:
        x_vals = original_df['x']
        if len(x_vals) > 0:
            x_min, x_max = min(x_vals), max(x_vals)
            span = x_max - x_min if x_max != x_min else 1.0
            delta = span * 0.15  # Length of tangent visual relative to domain
            
            for i, row in original_df.iterrows():
                xi = row['x']
                yi = row['y']
                mi = row['yp']
                
                # Calculate tangent segment endpoints
                xt = [xi - delta/2, xi + delta/2]
                yt = [yi - mi*(delta/2), yi + mi*(delta/2)]
                
                fig.add_trace(go.Scatter(
                    x=xt, y=yt,
                    mode='lines',
                    line=dict(color='#f59e0b', width=2, dash='dot'),
                    name='Tangents' if i == 0 else None,
                    legendgroup='tangents',
                    showlegend=(i == 0),
                    hoverinfo='skip'
                ))

    # 4. Plot The Original Data Points
    fig.add_trace(go.Scatter(
        x=original_df['x'], y=original_df['y'], 
        mode='markers', name='Data Points',
        marker=dict(color='#ef4444', size=10, symbol='circle', line=dict(color='white', width=2))
    ))
    
    # 5. Highlight Evaluated Point
    if eval_pt:
        val_in, val_out = eval_pt
        # If inverse: val_in is Y, val_out is X
        if inverse:
            pt_x, pt_y = val_out, val_in
            hover_txt = f"Evaluated: x({val_in}) = {val_out:.4f}"
        else:
            pt_x, pt_y = val_in, val_out
            hover_txt = f"Evaluated: f({val_in}) = {val_out:.4f}"
            
        fig.add_trace(go.Scatter(
            x=[pt_x], y=[pt_y],
            mode='markers', name='Evaluated Point',
            marker=dict(color='#10b981', size=10, symbol='diamond', line=dict(color='white', width=2)),
            text=[hover_txt],
            hoverinfo="text"
        ))
    
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="x", yaxis_title="y",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)