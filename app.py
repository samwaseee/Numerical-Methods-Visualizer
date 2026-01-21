import streamlit as st
import plotly.graph_objects as go
import numpy as np
from components.navbar import render_navbar
import views.input_page as input_view
import views.result_page as result_view
import views.dashboard_page as dashboard_view
import views.interpolation_page as interpolation_view
import views.differentiation_page as differentiation_view

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Numerical Methods Visualizer", page_icon="logo.svg")

# --- 2. INITIALIZE STATE ---
input_view.initialize_session_state()

# --- 4. NAVIGATION ROUTER ---
render_navbar()
header_container = st.empty()

def render_landing_page():
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 50px 0;">
            <p style="font-size: 1.2rem; color: #888; margin-bottom: 3rem;">Interactive tools for mathematical analysis</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # --- SECTION 1: ROOT FINDER ---
    with st.container(border=True):
        c_text, c_viz = st.columns([1, 1.5], gap="large")
        with c_text:
            st.subheader("Root Finder")
            st.markdown("Solve non-linear equations $f(x)=0$ using various iterative methods like Newton-Raphson, Bisection, and more.")
            st.markdown("""
            **Capabilities:**
            - Visualize convergence steps
            - Compare multiple methods
            - Analyze polynomial roots
            """)
            st.write("")
            if st.button("Launch Solver", type="primary", icon=":material/function:"):
                st.session_state.page = "input"
                st.query_params["page"] = "input"
                st.rerun()
        
        with c_viz:
            # Mini Graph for Root Finder
            x = np.linspace(-1, 4, 100)
            y = x**2 - 2*x - 3
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#4EA5FF', width=3)))
            fig.add_trace(go.Scatter(x=[3], y=[0], mode='markers', marker=dict(color='#ef4444', size=10, line=dict(color='white', width=2))))
            fig.add_annotation(x=3, y=0, text="Root", showarrow=True, arrowhead=1, ax=0, ay=-40)
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0), 
                height=200, 
                xaxis=dict(showgrid=False, zeroline=True),
                yaxis=dict(showgrid=False, zeroline=True, showticklabels=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    st.write("")

    # --- SECTION 2: INTERPOLATION ---
    with st.container(border=True):
        c_text, c_viz = st.columns([1, 1.5], gap="large")
        with c_text:
            st.subheader("Interpolation")
            st.markdown("Construct new data points within the range of a discrete set of known data points.")
            st.markdown("""
            **Capabilities:**
            - Lagrange & Newton Polynomials
            - Forward/Backward/Central Differences
            - Curve fitting visualization
            """)
            st.write("")
            if st.button("Launch Interpolation", type="primary", icon=":material/timeline:"):
                st.session_state.page = "interpolation"
                st.query_params["page"] = "interpolation"
                st.rerun()
        
        with c_viz:
            # Mini Graph for Interpolation
            x_pts = np.array([0, 1, 2, 3])
            y_pts = np.array([1, 3, 2, 4])
            x_line = np.linspace(-0.2, 3.2, 100)
            # Simple polyfit for visual
            z = np.polyfit(x_pts, y_pts, 3)
            p = np.poly1d(z)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', line=dict(color='#A855F7', width=3, dash='dash')))
            fig.add_trace(go.Scatter(x=x_pts, y=y_pts, mode='markers', marker=dict(color='#A855F7', size=10, line=dict(color='white', width=2))))
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0), 
                height=200, 
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    st.write("")

    # --- SECTION 3: DIFFERENTIATION ---
    with st.container(border=True):
        c_text, c_viz = st.columns([1, 1.5], gap="large")
        with c_text:
            st.subheader("Differentiation")
            st.markdown("Calculate derivatives from tabular data using finite difference methods.")
            st.markdown("""
            **Capabilities:**
            - 1st & 2nd Derivatives
            - Newton's Forward/Backward
            - Tangent Line Visualization
            """)
            st.write("")
            if st.button("Launch Differentiation", type="primary", icon=":material/show_chart:"):
                st.session_state.page = "differentiation"
                st.query_params["page"] = "differentiation"
                st.rerun()
        
        with c_viz:
            # Mini Graph for Differentiation
            x = np.linspace(0, 4, 100); y = np.sin(x)
            y_tan = np.sin(2) + np.cos(2) * (x - 2)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#F59E0B', width=3)))
            fig.add_trace(go.Scatter(x=x, y=y_tan, mode='lines', line=dict(color='#10B981', width=2, dash='dot')))
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0), height=200, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

current_page = st.session_state.page

if current_page == "landing":
    render_landing_page()
elif current_page == "input":
    input_view.show_input_page(header_container)
elif current_page == "result":
    result_view.show_result_page(header_container)
elif current_page == "dashboard":
    dashboard_view.show_dashboard_page(header_container)
elif current_page == "interpolation":
    interpolation_view.show_interpolation_page()
elif current_page == "differentiation":
    differentiation_view.show_differentiation_page()

# Sync Page to URL
if st.query_params.get("page") != current_page:
    st.query_params["page"] = current_page