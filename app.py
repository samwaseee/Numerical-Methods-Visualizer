import streamlit as st
import plotly.graph_objects as go
import numpy as np
from components.navbar import render_navbar
import views.input_page as input_view
import views.result_page as result_view
import views.dashboard_page as dashboard_view
import views.interpolation_page as interpolation_view
import views.differentiation_page as differentiation_view
import views.integration_page as integration_view
import views.fitting_page as fitting_view
import views.linear_page as linear_view
import views.ode_page as ode_view

# --- 1. CONFIGURATION ---
st.set_page_config(
    layout="wide", page_title="Numerical Methods Visualizer", page_icon="logo.svg"
)

# --- 2. INITIALIZE STATE ---
input_view.initialize_session_state()

# --- 3. DARK MODE SETTINGS ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            [data-testid="stSidebar"] {
                background-color: #262730;
            }
            [data-testid="stHeader"] {
                background-color: rgba(0,0,0,0);
            }
            .stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div {
                background-color: #1F2937;
                color: #FAFAFA;
            }
            h1, h2, h3, h4, h5, h6, label {
                color: #FAFAFA !important;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

# --- 4. NAVIGATION ROUTER ---
render_navbar()
header_container = st.empty()


def render_landing_page():
    # --- HERO SECTION ---
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 4rem 0;">
            <h1 style="font-size: 3.5rem; font-weight: 700; margin-bottom: 1rem; background: linear-gradient(90deg, #4EA5FF, #A855F7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Numerical Methods Visualizer</h1>
            <p style="font-size: 1.2rem; color: #888; max-width: 700px; margin: 0 auto; line-height: 1.6;">
                Explore, visualize, and understand complex numerical algorithms. 
                From finding roots to solving differential equations, get step-by-step insights into the math.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- VISUALIZATION HELPERS ---
    def viz_root():
        x = np.linspace(-1, 4, 100)
        y = x**2 - 2 * x - 3
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines", name="f(x)", line=dict(color="#4EA5FF", width=3)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[3],
                y=[0],
                mode="markers",
                name="Root",
                marker=dict(
                    color="#ef4444", size=10, line=dict(color="white", width=2)
                ),
            )
        )
        fig.add_annotation(
            x=3,
            y=0,
            text="Root (x=3)",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color="#FAFAFA"),
        )
        fig.update_layout(
            showlegend=False,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=180,
            xaxis=dict(showgrid=False, zeroline=True, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=True, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    def viz_linear():
        x = np.linspace(-2, 2, 15)
        y = np.linspace(-2, 2, 15)
        X, Y = np.meshgrid(x, y)

        Z1 = X + 2 * Y
        Z2 = X - Y
        Z3 = X + Y

        fig = go.Figure()

        fig.add_trace(
            go.Surface(
                z=Z1,
                x=X,
                y=Y,
                colorscale="Blues",
                showscale=False,
                opacity=0.7,
                name="Plane 1",
            )
        )
        fig.add_trace(
            go.Surface(
                z=Z2,
                x=X,
                y=Y,
                colorscale="Purples",
                showscale=False,
                opacity=0.7,
                name="Plane 2",
            )
        )
        fig.add_trace(
            go.Surface(
                z=Z3,
                x=X,
                y=Y,
                colorscale="Greens",
                showscale=False,
                opacity=0.7,
                name="Plane 3",
            )
        )

        # 4. Add the Intersection point (Visual aid)
        # Based on these equations, the intersection is indeed at (0,0,0)
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                marker=dict(color="#ef4444", size=5, symbol="diamond"),
                name="Intersection",
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=True, title="x", showticklabels=False),
                yaxis=dict(visible=True, title="y", showticklabels=False),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectratio=dict(x=1, y=1, z=1.6),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=365,
            showlegend=True,
            legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True, 
                        config={"displayModeBar": False,
                                "scrollZoom": False,
                                "editable": False,
                                "showLink": False
                                })

    def viz_interp():
        x_pts = np.array([0, 1, 2, 3])
        y_pts = np.array([1, 3, 2, 4])
        x_line = np.linspace(-0.2, 3.2, 100)
        z = np.polyfit(x_pts, y_pts, 3)
        p = np.poly1d(z)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=p(x_line),
                mode="lines",
                name="Polynomial",
                line=dict(color="#A855F7", width=3, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="markers",
                name="Points",
                marker=dict(
                    color="#A855F7", size=10, line=dict(color="white", width=2)
                ),
            )
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=180,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    def viz_fitting():
        x = np.linspace(0, 5, 20)
        y_noise = 0.5 * x + 1 + np.random.normal(0, 0.2, 20)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_noise,
                mode="markers",
                name="Data",
                marker=dict(color="#ef4444", size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=0.5 * x + 1,
                mode="lines",
                name="Fit",
                line=dict(color="#6366f1", width=2),
            )
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=180,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    def viz_diff():
        x = np.linspace(0, 4, 100)
        y = np.sin(x)
        y_tan = np.sin(2) + np.cos(2) * (x - 2)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines", name="f(x)", line=dict(color="#F59E0B", width=3)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_tan,
                mode="lines",
                name="Tangent",
                line=dict(color="#10B981", width=2, dash="dot"),
            )
        )
        fig.add_annotation(
            x=2,
            y=np.sin(2),
            text="Tangent (x=2)",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color="#FAFAFA"),
        )
        fig.update_layout(
            showlegend=False,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=180,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    def viz_int():
        x = np.linspace(0, 3, 100)
        y = x**2 * np.exp(-x)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="tozeroy",
                mode="lines",
                name="Area",
                line=dict(color="#8B5CF6", width=3),
            )
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(x=0.6, y=0.98, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=180,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    def viz_ode():
        # 1. Define the Domain and Function
        x = np.linspace(0, 4, 100)
        y = np.exp(0.5 * x) # Solution to y' = 0.5y

        fig = go.Figure()

        # 2. Main Solution Curve (with Area Fill)
        fig.add_trace(
            go.Scatter(
                x=x, y=y, 
                mode="lines", 
                name="y(x)", 
                line=dict(color="#10b981", width=3),
                fill='tozeroy', # Adds the nice area under the curve
                fillcolor='rgba(16, 185, 129, 0.1)' # Transparent Green
            )
        )

        # 3. Add "Slope Field" Arrows (Mathematically Accurate)
        # We sample fewer points to avoid clutter
        sample_x = np.linspace(0.5, 3.5, 5) 
        sample_y = np.exp(0.5 * sample_x)
        slopes = 0.5 * sample_y # The derivative dy/dx

        for i in range(len(sample_x)):
            # Calculate arrow direction based on slope
            # Visual scaling factor for the arrow tail
            scale = 20 
            # Note: In pixel coords, +y is DOWN. So a positive slope (Up-Right) 
            # means the tail must be "Down-Left" (ax < 0, ay > 0).

            # We normalize slightly to keep arrows consistent length visually
            m = slopes[i]
            norm = np.sqrt(1 + m**2)
            ax_px = -scale * (1/norm) 
            ay_px = scale * (m/norm) 

            fig.add_annotation(
                x=sample_x[i],
                y=sample_y[i],
                ax=ax_px, # Tail X offset (pixels)
                ay=ay_px, # Tail Y offset (pixels)
                xref="x", yref="y",
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#374151" # Dark gray for contrast
            )

        # 4. Polish the Layout
        fig.update_layout(
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=150,
            xaxis=dict(
                showgrid=True, 
                gridcolor='rgba(128,128,128,0.2)', 
                showticklabels=False, 
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='rgba(128,128,128,0.2)', 
                showticklabels=False, 
                zeroline=False
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    # --- SECTION 1: EQUATION SOLVERS ---
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#4EA5FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><line x1="8" y1="12" x2="16" y2="12"></line><line x1="12" y1="8" x2="12" y2="16"></line></svg>
            <h3 style="margin: 0; padding: 0;">Equation Solvers</h3>
        </div>
        <hr style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.container(border=True):
            st.subheader("Root Finder")
            st.markdown(
                "Solve non-linear equations $f(x)=0$. Visualize how different algorithms converge to the root."
            )
            st.markdown(
                """
            **Available Methods:**
            - **Bracketing:** Bisection, False Position
            - **Open:** Newton-Raphson, Secant, Fixed Point
            - **Advanced:** Brent's, Halley's, Muller's
            """
            )
            viz_root()
            if st.button(
                "Launch Root Finder", type="primary", use_container_width=True
            ):
                st.session_state.page = "input"
                st.query_params["page"] = "input"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.subheader("Linear Systems")
            lin_col1, lin_col2 = st.columns([1, 1])
            with lin_col1:
                st.markdown(
                    "Solve systems of linear equations $Ax=B$ using direct and iterative techniques."
                )
                st.markdown(
                    """
                **Available Methods:**
                - **Direct:** Gauss Elimination, Gauss-Jordan, LU, Cholesky, Cramer's
                - **Iterative:** Jacobi, Gauss-Seidel
                - **Matrix Ops:** Inversion, Determinant
                """
                )
            with lin_col2:
                viz_linear()
            if st.button(
                "Launch Linear Solver", type="primary", use_container_width=True
            ):
                st.session_state.page = "linear"
                st.query_params["page"] = "linear"
                st.rerun()

    st.write("")

    # --- SECTION 2: DATA ANALYSIS ---
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#A855F7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            <h3 style="margin: 0; padding: 0;">Data Analysis</h3>
        </div>
        <hr style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
        """,
        unsafe_allow_html=True,
    )

    col3, col4 = st.columns(2, gap="medium")

    with col3:
        with st.container(border=True):
            st.subheader("Interpolation")
            st.markdown(
                "Estimate unknown values between known data points using polynomial techniques."
            )
            st.markdown(
                """
            **Available Methods:**
            - **Polynomial:** Lagrange, Newton Divided Difference
            - **Finite Diff:** Newton Forward/Backward, Gauss, Stirling, Bessel
            - **Splines:** Cubic Spline, Hermite
            """
            )
            viz_interp()
            if st.button(
                "Launch Interpolation", type="primary", use_container_width=True
            ):
                st.session_state.page = "interpolation"
                st.query_params["page"] = "interpolation"
                st.rerun()

    with col4:
        with st.container(border=True):
            st.subheader("Curve Fitting")
            st.markdown(
                "Find the best-fitting curve for a dataset using regression analysis."
            )
            st.markdown(
                """
            **Available Methods:**
            - **Linear Regression:** Simple $y=mx+c$
            - **Polynomial Regression:** High-degree polynomials
            - **Exponential Fit:** Non-linear growth models
            """
            )
            viz_fitting()
            if st.button(
                "Launch Curve Fitting", type="primary", use_container_width=True
            ):
                st.session_state.page = "fitting"
                st.query_params["page"] = "fitting"
                st.rerun()

    st.write("")

    # --- SECTION 3: CALCULUS ---
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 18c0-2.5 2-5 5-5s5 2.5 5 5c0 2.5 2 5 5 5s5-2.5 5-5"></path></svg>
            <h3 style="margin: 0; padding: 0;">Calculus</h3>
        </div>
        <hr style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
        """,
        unsafe_allow_html=True,
    )

    col5, col6, col7 = st.columns(3, gap="medium")

    with col5:
        with st.container(border=True):
            st.subheader("Differentiation")
            st.markdown("Compute derivatives from tabular data.")
            st.markdown(
                """
            **Methods:**
            - Finite Differences (Fwd/Bwd/Cen)
            - Richardson Extrapolation
            - Derivative via Splines
            """
            )
            viz_diff()
            if st.button(
                "Launch Differentiation", type="primary", use_container_width=True
            ):
                st.session_state.page = "differentiation"
                st.query_params["page"] = "differentiation"
                st.rerun()

    with col6:
        with st.container(border=True):
            st.subheader("Integration")
            st.markdown("Numerical approximation of definite integrals.")
            st.markdown(
                """
            **Methods:**
            - Trapezoidal Rule
            - Simpson's 1/3 & 3/8
            - Boole's & Weddle's Rules
            """
            )
            viz_int()
            if st.button(
                "Launch Integration", type="primary", use_container_width=True
            ):
                st.session_state.page = "integration"
                st.query_params["page"] = "integration"
                st.rerun()

    with col7:
        with st.container(border=True):
            st.subheader("Ordinary Differential Equations")
            st.markdown("Solve Initial Value Problems (IVP).")
            st.markdown(
                """
            **Methods:**
            - Euler & Heun
            - Runge-Kutta (RK4)
            - Adaptive RK45
            - Systems of ODEs
            """
            )
            viz_ode()
            if st.button("Launch ODE Solver", type="primary", use_container_width=True):
                st.session_state.page = "ode"
                st.query_params["page"] = "ode"
                st.rerun()

    st.write("")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 20px;">
            <small>Built with Streamlit & Plotly • Numerical Methods Project</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
elif current_page == "integration":
    integration_view.show_integration_page()
elif current_page == "fitting":
    fitting_view.show_fitting_page()
elif current_page == "linear":
    linear_view.show_linear_page()
elif current_page == "ode":
    ode_view.show_ode_page()

# Sync Page to URL
if st.query_params.get("page") != current_page:
    st.query_params["page"] = current_page
