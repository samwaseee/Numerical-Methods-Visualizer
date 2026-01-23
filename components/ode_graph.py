import plotly.graph_objects as go
import numpy as np
import streamlit as st

def render_ode_chart(results_dict, f_func):
    st.subheader("Solution Trajectories & Slope Field")
    
    if not results_dict:
        st.warning("No data to plot.")
        return

    fig = go.Figure()
    
    # --- 1. Slope Field (Quiver-like) ---
    # Determine plot boundaries based on all data
    all_x = []
    all_y = []
    for method, df in results_dict.items():
        all_x.extend(df['x'].tolist())
        all_y.extend(df['y'].tolist())
    
    if not all_x: return
    x_min, x_max = min(all_x), max(all_x)
    
    y_min, y_max = min(all_y), max(all_y)
    
    # Add padding for the grid
    span_x = x_max - x_min if x_max != x_min else 1.0
    span_y = y_max - y_min if y_max != y_min else 1.0
    
    grid_x_min, grid_x_max = x_min - span_x*0.1, x_max + span_x*0.1
    grid_y_min, grid_y_max = y_min - span_y*0.1, y_max + span_y*0.1
    
    # Create grid
    num_arrows = 20
    x_grid = np.linspace(grid_x_min, grid_x_max, num_arrows)
    y_grid = np.linspace(grid_y_min, grid_y_max, num_arrows)
    
    # Scale for arrow length (visual only)
    scale_factor = min(span_x, span_y) / num_arrows * 0.6
    
    lines_x = []
    lines_y = []
    
    for x in x_grid:
        for y in y_grid:
            try:
                slope = f_func(x, y)
                # Normalize the vector (1, slope) to make arrows uniform length
                # Vector V = [1, slope]
                norm = np.sqrt(1 + slope**2)
                if norm == 0: continue
                
                dx = (1 / norm) * scale_factor
                dy = (slope / norm) * scale_factor
                
                # Line segment centered at (x,y)
                lines_x.extend([x - dx, x + dx, None])
                lines_y.extend([y - dy, y + dy, None])
            except:
                pass

    fig.add_trace(go.Scatter(
        x=lines_x, y=lines_y,
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
        name='Slope Field',
        hoverinfo='skip'
    ))

    # --- 2. Solution Curves ---
    colors = {'Euler': '#ef4444', 'Heun': '#f59e0b', 'RK4': '#10b981', 'RK45': '#8b5cf6'}
    
    for method, df in results_dict.items():
        color = colors.get(method, '#3b82f6')
        
        # Show markers for RK45 to visualize adaptive steps
        mode = 'lines+markers' if method == 'RK45' else 'lines'
        marker_size = 4 if method == 'RK45' else 0
        
        fig.add_trace(go.Scatter(
            x=df['x'], y=df['y'],
            mode=mode,
            name=method,
            line=dict(color=color, width=2),
            marker=dict(size=marker_size)
        ))

    fig.update_layout(
        title="ODE Solution",
        xaxis_title="x", yaxis_title="y",
        height=550,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_system_ode_chart(df, indep_var, dep_vars):
    st.subheader("System Solution Visualization")
    
    tab1, tab2 = st.tabs(["Time Series", "Phase Portrait"])
    
    with tab1:
        fig_ts = go.Figure()
        colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
        
        for i, var in enumerate(dep_vars):
            fig_ts.add_trace(go.Scatter(
                x=df[indep_var], 
                y=df[var],
                mode='lines',
                name=var,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
            
        fig_ts.update_layout(
            title="Time Series",
            xaxis_title=indep_var,
            yaxis_title="Values",
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
    with tab2:
        if len(dep_vars) >= 2:
            # --- ANIMATION CONTROLS ---
            n_points = len(df)
            if n_points > 1:
                t_idx = st.slider("Time Animation (t)", 0, n_points - 1, 0, format="")
                
                curr_t = df[indep_var].iloc[t_idx]
                curr_x = df[dep_vars[0]].iloc[t_idx]
                curr_y = df[dep_vars[1]].iloc[t_idx]
                
                c_val1, c_val2 = st.columns(2)
                c_val1.metric(f"{indep_var}", f"{curr_t:.2f}")
                c_val2.metric(f"State ({dep_vars[0]}, {dep_vars[1]})", f"({curr_x:.2f}, {curr_y:.2f})")
            else:
                t_idx = 0
                curr_x, curr_y = 0, 0

            fig_pp = go.Figure()
            
            # Full Trajectory
            fig_pp.add_trace(go.Scatter(
                x=df[dep_vars[0]],
                y=df[dep_vars[1]],
                mode='lines',
                name=f"{dep_vars[1]} vs {dep_vars[0]}",
                line=dict(color='#6366f1', width=2)
            ))
            
            # Current State Marker
            if n_points > 1:
                fig_pp.add_trace(go.Scatter(
                    x=[curr_x], y=[curr_y],
                    mode='markers',
                    name='Current State',
                    marker=dict(color='#ef4444', size=12, line=dict(color='white', width=2))
                ))

            fig_pp.update_layout(
                title=f"Phase Portrait ({dep_vars[0]} vs {dep_vars[1]})",
                xaxis_title=dep_vars[0],
                yaxis_title=dep_vars[1],
                height=500
            )
            st.plotly_chart(fig_pp, use_container_width=True)
        else:
            st.info("Phase portrait requires at least 2 dependent variables.")
