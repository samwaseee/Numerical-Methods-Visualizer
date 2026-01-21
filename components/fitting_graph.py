import plotly.graph_objects as go
import streamlit as st
import numpy as np

def render_fitting_chart(x_data, y_data, result):
    st.subheader("Visualizing the Fit")
    
    fig = go.Figure()
    
    # 1. Plot Original Data (Scatter)
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        mode='markers',
        name='Noisy Data',
        marker=dict(color='#ef4444', size=10, line=dict(color='white', width=2))
    ))
    
    # 2. Plot Fitted Curve
    if result and "predict" in result:
        func = result["predict"]
        
        # Generate smooth line
        x_min, x_max = min(x_data), max(x_data)
        span = x_max - x_min if x_max != x_min else 1.0
        x_smooth = np.linspace(x_min - span*0.1, x_max + span*0.1, 200)
        y_smooth = func(x_smooth)
        
        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_smooth,
            mode='lines',
            name='Fitted Model',
            line=dict(color='#6366f1', width=3)
        ))
        
        # 3. Plot Residuals (Vertical lines)
        # We create a single trace with None to break the lines for efficiency
        y_pred_points = func(np.array(x_data))
        
        res_x = []
        res_y = []
        
        for i in range(len(x_data)):
            res_x.extend([x_data[i], x_data[i], None])
            res_y.extend([y_data[i], y_pred_points[i], None])
            
        fig.add_trace(go.Scatter(
            x=res_x, y=res_y,
            mode='lines',
            name='Residuals',
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5,
            legendgroup="residuals"
        ))

    fig.update_layout(
        title=f"Curve Fitting: {result.get('type', '')}",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="x", yaxis_title="y",
        hovermode="closest"
    )
    
    st.plotly_chart(fig, use_container_width=True)
