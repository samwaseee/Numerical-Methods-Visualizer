import plotly.graph_objects as go
import streamlit as st

def render_integration_chart(x_data, y_data, method):
    st.subheader("Visualizing the Area")
    
    fig = go.Figure()
    
    # Fill area
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        fill='tozeroy',
        mode='lines+markers',
        name='f(x)',
        line=dict(color='#8b5cf6', width=2),
        marker=dict(size=6, color='#7c3aed'),
        fillcolor='rgba(139, 92, 246, 0.2)' # Light purple fill
    ))
    
    # Zero Line
    fig.add_hline(y=0, line_color="gray", line_width=1)
    
    fig.update_layout(
        title=f"Area under curve ({method})",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="x", yaxis_title="y",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
