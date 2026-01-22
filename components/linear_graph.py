import plotly.graph_objects as go
import numpy as np
import streamlit as st

def render_linear_system_graph(A, B, solution=None):
    """
    Visualizes 2D lines or 3D planes for a linear system Ax = B.
    """
    # Ensure inputs are numpy arrays
    A = np.array(A)
    B = np.array(B)
    
    rows, cols = A.shape
    
    if rows != cols:
        # Visualization is primarily for square systems in this context
        return

    if rows == 2:
        _render_2d_system(A, B, solution)
    elif rows == 3:
        _render_3d_system(A, B, solution)
    else:
        st.info("Visualization is available for 2D and 3D systems only.")

def _render_2d_system(A, B, solution):
    fig = go.Figure()
    
    # Determine plot range
    if solution is not None:
        center_x = float(solution[0])
        span = 10
    else:
        center_x = 0
        span = 10
        
    x_range = np.linspace(center_x - span, center_x + span, 100)
    colors = ['#6366f1', '#10b981'] # Purple, Emerald

    for i in range(2):
        a1, a2 = A[i]
        b = B[i]
        
        # Equation: a1*x + a2*y = b
        
        if abs(a2) < 1e-6:
            # Vertical line: a1*x = b => x = b/a1
            if abs(a1) > 1e-6:
                x_val = b / a1
                # Create a vertical line segment
                y_min = (float(solution[1]) - 10) if solution is not None else -10
                y_max = (float(solution[1]) + 10) if solution is not None else 10
                
                fig.add_trace(go.Scatter(
                    x=[x_val, x_val],
                    y=[y_min, y_max],
                    mode='lines',
                    name=f'Eq {i+1}',
                    line=dict(color=colors[i%len(colors)], width=2),
                    hovertemplate=f"{a1:.2f}x = {b:.2f}<extra></extra>"
                ))
        else:
            # y = (b - a1*x) / a2
            y_vals = (b - a1 * x_range) / a2
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_vals,
                mode='lines',
                name=f'Eq {i+1}',
                line=dict(color=colors[i%len(colors)], width=2),
                hovertemplate=f"{a1:.2f}x + {a2:.2f}y = {b:.2f}<extra></extra>"
            ))

    # Plot Intersection Point
    if solution is not None:
        fig.add_trace(go.Scatter(
            x=[float(solution[0])],
            y=[float(solution[1])],
            mode='markers',
            name='Solution',
            marker=dict(color='#ef4444', size=12, symbol='diamond', line=dict(color='white', width=2)),
            hovertemplate=f"({solution[0]:.2f}, {solution[1]:.2f})<extra></extra>"
        ))

    fig.update_layout(
        title="2D Linear System Visualization",
        xaxis_title="x",
        yaxis_title="y",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True)

def _render_3d_system(A, B, solution):
    fig = go.Figure()
    
    # Determine range
    if solution is not None:
        center = [float(s) for s in solution]
        span = 5
    else:
        center = [0, 0, 0]
        span = 10
        
    grid_points = 20
    colors = ['#6366f1', '#10b981', '#f59e0b'] # Purple, Emerald, Amber
    
    for i in range(3):
        a1, a2, a3 = A[i]
        b = B[i]
        
        # Format equation string for hover
        eq_str = f"{a1:.2f}x + {a2:.2f}y + {a3:.2f}z = {b:.2f}".replace("+ -", "- ")
        
        # We need to solve for one variable to plot a surface.
        # Priority: z, then y, then x (to handle planes parallel to axes)
        
        if abs(a3) > 1e-6:
            # Solve for z: z = (b - a1*x - a2*y) / a3
            x = np.linspace(center[0] - span, center[0] + span, grid_points)
            y = np.linspace(center[1] - span, center[1] + span, grid_points)
            X, Y = np.meshgrid(x, y)
            Z = (b - a1 * X - a2 * Y) / a3
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                opacity=0.5,
                colorscale=[[0, colors[i]], [1, colors[i]]],
                showscale=False,
                name=f'Plane {i+1}',
                hovertemplate=f"<b>Plane {i+1}</b><br>{eq_str}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>"
            ))
            
        elif abs(a2) > 1e-6:
            # Solve for y: y = (b - a1*x - a3*z) / a2
            x = np.linspace(center[0] - span, center[0] + span, grid_points)
            z = np.linspace(center[2] - span, center[2] + span, grid_points)
            X, Z = np.meshgrid(x, z)
            Y = (b - a1 * X - a3 * Z) / a2
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                opacity=0.5,
                colorscale=[[0, colors[i]], [1, colors[i]]],
                showscale=False,
                name=f'Plane {i+1}',
                hovertemplate=f"<b>Plane {i+1}</b><br>{eq_str}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>"
            ))
            
        elif abs(a1) > 1e-6:
            # Solve for x: x = (b - a2*y - a3*z) / a1
            y = np.linspace(center[1] - span, center[1] + span, grid_points)
            z = np.linspace(center[2] - span, center[2] + span, grid_points)
            Y, Z = np.meshgrid(y, z)
            X = (b - a2 * Y - a3 * Z) / a1
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                opacity=0.5,
                colorscale=[[0, colors[i]], [1, colors[i]]],
                showscale=False,
                name=f'Plane {i+1}',
                hovertemplate=f"<b>Plane {i+1}</b><br>{eq_str}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>"
            ))
        else:
            # Degenerate equation (0 = b), skip visualization
            continue

    # Plot Intersection Point
    if solution is not None:
        fig.add_trace(go.Scatter3d(
            x=[float(solution[0])],
            y=[float(solution[1])],
            z=[float(solution[2])],
            mode='markers',
            name='Solution',
            marker=dict(color='#ef4444', size=8, symbol='diamond', line=dict(color='white', width=2)),
            hovertemplate=f"<b>Solution</b><br>x: {solution[0]:.2f}<br>y: {solution[1]:.2f}<br>z: {solution[2]:.2f}<extra></extra>"
        ))

    fig.update_layout(
        title="3D Linear System Visualization",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z'
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
