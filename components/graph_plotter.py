import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def render_scanner_chart(solver, a, b, step, roots, scan_log):
    # 1. Prepare Data
    x_vals = np.linspace(a, b, 500)
    try:
        y_vals = solver.f(x_vals)
    except:
        y_vals = np.zeros_like(x_vals)

    # 2. Create Figure
    fig = go.Figure()

    # Main Function Line
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name='f(x)', line=dict(color='#4EA5F9', width=2)))
    
    # Zero Line (Axis)
    fig.add_hline(y=0, line_color="gray", line_width=3)
    fig.add_vline(x=0, line_color="gray", line_width=3)

    # Roots Markers
    if roots:
        fig.add_trace(go.Scatter(
            x=roots, 
            y=[0]*len(roots), 
            mode='markers', 
            name='Roots', 
            marker=dict(color='#F43F5E', size=10, symbol='circle-open', line=dict(width=2))
        ))

    # 3. INTERACTIVE SLIDER (The "Scanner" visual)
    if scan_log:
        df_log = pd.DataFrame(scan_log)
        
        # Slider to step through the log
        step_idx = st.slider(
            "Select Interval", 
            min_value=0, 
            max_value=len(df_log)-1, 
            value=0,
            format="Step %d"
        )
        
        # Get current step data using NEW KEY NAMES
        current_row = df_log.iloc[step_idx]
        
        x_curr = current_row['Interval Start']
        x_next = current_row['Interval End']

        # Highlight the current interval being scanned
        fig.add_vrect(
            x0=x_curr, x1=x_next, 
            fillcolor="yellow", opacity=0.2, 
            line_width=0, 
            annotation_text="Scanning...", annotation_position="top left"
        )
        
        # Add a point for where we are checking
        fig.add_trace(go.Scatter(
            x=[x_curr], y=[solver.f(x_curr)],
            mode='markers', name='Check',
            marker=dict(color='yellow', size=8)
        ))

        fig.update_layout(
            title="Root Scanning Visualization",
            xaxis_title="x", yaxis_title="f(x)",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return current_row, step_idx

    else:
        st.plotly_chart(fig, use_container_width=True)
        return None, 0

def render_iteration_chart(solver, history, method_name, g_str=None):
    df = pd.DataFrame(history)
    
    # Handle single-step iteration slider bug
    max_step = len(df)
    
    if max_step > 1:
        step_idx = st.slider("Step through iterations", 1, max_step, max_step)
    else:
        step_idx = 1
        st.info("Converged in a single step!")
    
    # Slice data up to current step
    current_df = df.iloc[:step_idx]
    last_row = current_df.iloc[-1]
    
    # Determine bounds for plotting
    try:
        x_points = []
        if 'x_new' in df.columns: x_points.extend(df['x_new'].tolist())
        if 'x_old' in df.columns: x_points.extend(df['x_old'].tolist())
        if 'x_prev' in df.columns: x_points.extend(df['x_prev'].tolist())
        if 'x_curr' in df.columns: x_points.extend(df['x_curr'].tolist())
        if 'a' in df.columns: x_points.extend(df['a'].tolist())
        if 'b' in df.columns: x_points.extend(df['b'].tolist())
        
        if not x_points: x_points = [-10, 10]
        
        min_x, max_x = min(x_points), max(x_points)
        span = max_x - min_x if max_x != min_x else 2.0
        plot_min, plot_max = min_x - span*0.2, max_x + span*0.2
    except:
        plot_min, plot_max = -10, 10

    # Generate Curve
    x_range = np.linspace(plot_min, plot_max, 400)
    try:
        y_range = solver.f(x_range)
    except:
        y_range = np.zeros_like(x_range)

    fig = go.Figure()
    
    # 1. Base Function
    fig.add_trace(go.Scatter(x=x_range, y=y_range, name='f(x)', line=dict(color="#c3c8f6", width=3)))
    fig.add_hline(y=0, line_color="gray", opacity=0.5)

    # 2. Method Specific Visuals
    
    # A. Fixed Point (Cobweb Plot)
    if method_name == "Fixed Point Iteration" and g_str:
        fig.add_trace(go.Scatter(x=x_range, y=x_range, name='y=x', line=dict(color='gray', dash='dash')))
        
        path_x = []
        path_y = []
        
        if not current_df.empty:
            x_start = current_df.iloc[0]['x_old']
            path_x.append(x_start); path_y.append(0)
            path_x.append(x_start); path_y.append(current_df.iloc[0]['g(x)']) 
            
            for i, row in current_df.iterrows():
                val = row['x_new']
                path_x.append(val); path_y.append(val)
                if i < len(current_df)-1:
                    path_x.append(val); path_y.append(current_df.iloc[i+1]['g(x)'])

        fig.add_trace(go.Scatter(
            x=path_x, y=path_y, 
            mode='lines+markers', 
            name='Iter Path',
            line=dict(color='#ec4899', width=1.5),
            marker=dict(size=4)
        ))

    # B. Newton Raphson (Tangent Lines)
    elif method_name == "Newton-Raphson":
        for i, row in current_df.iterrows():
            x_old = row['x_old']
            fx = row['f(x)']
            x_new = row['x_new']
            
            fig.add_trace(go.Scatter(
                x=[x_old, x_old], y=[0, fx],
                mode='lines', line=dict(color='gray', dash='dot', width=1),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[x_old, x_new], y=[fx, 0],
                mode='lines', line=dict(color="#ee0d0d", width=2),
                opacity=0.5 if i < step_idx-1 else 1.0, 
                name='Tangent' if i == step_idx-1 else None
            ))

    # C. Secant Method (Secant Lines)
    elif method_name == "Secant Method":
        for i, row in current_df.iterrows():
            x0 = row['x_prev']
            x1 = row['x_curr']
            x_new = row['x_new']
            
            try:
                f0 = solver.f(x0)
                f1 = solver.f(x1)
                
                fig.add_trace(go.Scatter(
                    x=[x0, x1, x_new], y=[f0, f1, 0],
                    mode='lines+markers', line=dict(color='#f59e0b', width=1.5),
                    marker=dict(size=5),
                    opacity=0.5 if i < step_idx-1 else 1.0,
                    name='Secant' if i == step_idx-1 else None
                ))
            except: pass

    # D. Bracketing Methods (Bisection / False Position)
    elif method_name in ["Bisection Method", "False Position"]:
        curr_a = last_row['a']
        curr_b = last_row['b']
        
        # --- FIX: VISUALIZE BOUNDARIES ---
        # 1. Shaded Region
        fig.add_vrect(
            x0=curr_a, x1=curr_b,
            fillcolor="green", opacity=0.1,
            annotation_text="Active Bracket", annotation_position="top left"
        )
        
        # 2. Distinct Lines for A and B
        fig.add_vline(x=curr_a, line_width=1, line_dash="dash", line_color="white", annotation_text="a", annotation_position="bottom right")
        fig.add_vline(x=curr_b, line_width=1, line_dash="dash", line_color="white", annotation_text="b", annotation_position="bottom left")
        # ---------------------------------
        
        # Show calculated root estimate 'c'
        fig.add_trace(go.Scatter(
            x=[last_row['c (root)']], y=[0],
            mode='markers', marker=dict(color='red', size=10, symbol='x'),
            name='Current Root (c)'
        ))

    fig.update_layout(
        title=f"{method_name} (Step {step_idx})",
        xaxis_title="x", yaxis_title="f(x)",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)
    return last_row, step_idx