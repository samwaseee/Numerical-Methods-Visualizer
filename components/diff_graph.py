import plotly.graph_objects as go
import streamlit as st
import numpy as np
from algorithms_differentiation import DifferentiationSolver
from algorithms_interpolation import InterpolationSolver

def render_differentiation_chart(orig_data, target_x, res, method, extrema=None):
    """
    Renders the differentiation chart with curve, points, and tangent line.
    """
    # Visualization
    if "Divided Difference" in method:
        # Use InterpolationSolver for Newton DD
        solver = InterpolationSolver(orig_data['x'].values, orig_data['y'].values)
        solver.newton_divided_difference()
        interp_solver = solver
    else:
        solver = DifferentiationSolver(orig_data['x'].values, orig_data['y'].values)
        # Re-run purely for plotting context to set internal state of interpolation solver
        if "Forward" in method: solver.newton_forward_diff(target_x)
        elif "Backward" in method: solver.newton_backward_diff(target_x)
        elif "Central" in method: solver.central_diff(target_x)
        elif "Stirling" in method: solver.stirling_diff(target_x)
        elif "Bessel" in method: solver.bessel_diff(target_x)
        elif "Spline" in method: solver.cubic_spline_diff(target_x)
        else: solver.lagrange_derivative(target_x)
        interp_solver = solver.interp
    
    x_plot, y_plot = interp_solver.get_eval_points()
    
    fig = go.Figure()
    if len(x_plot) > 0:
        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='Curve', line=dict(color='#6366f1', width=3)))
    
    fig.add_trace(go.Scatter(x=orig_data['x'], y=orig_data['y'], mode='markers', name='Data', marker=dict(color='white', size=8, line=dict(color='black', width=1))))
    
    # Tangent
    span = (max(orig_data['x']) - min(orig_data['x'])) * 0.2 or 1.0
    x_tan = np.linspace(target_x - span, target_x + span, 100)
    y_tan = res['y'] + res['dy'] * (x_tan - target_x)
    
    fig.add_trace(go.Scatter(x=x_tan, y=y_tan, mode='lines', name=f"Tangent (Slope={res['dy']:.2f})", line=dict(color='#10b981', width=3, dash='dot')))
    fig.add_trace(go.Scatter(x=[target_x], y=[res['y']], mode='markers', name='Target', marker=dict(color='#10b981', size=8, symbol='diamond')))
    
    # --- Extrema Markers ---
    if extrema:
        for pt in extrema:
            color = "#ef4444" if pt['type'] == "Maxima" else "#3b82f6"
            
            fig.add_trace(go.Scatter(
                x=[pt['x']], 
                y=[pt['y']],
                mode='markers+text',
                name=pt['type'],
                text=[f"{pt['type']}"],
                textposition="top center",
                marker=dict(color=color, size=12, symbol="diamond", line=dict(width=2, color='white'))
            ))

    fig.update_layout(title="Geometric Visualization", height=450, xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, use_container_width=True)