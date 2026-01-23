import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
from algorithms_ode import ODESolver, SystemODESolver
from components.ode_graph import render_ode_chart, render_system_ode_chart
from components.ode_results import render_ode_steps

def show_ode_page():
    # --- NAVIGATION & CLEANUP ---
    if st.button("Back to Home", icon=":material/home:"):
        for key in list(st.session_state.keys()):
            if key != 'dark_mode':
                del st.session_state[key]
        st.query_params.clear()
        st.session_state.page = "landing"
        st.query_params["page"] = "landing"
        st.rerun()

    st.title("Ordinary Differential Equations (IVP)")
    st.markdown("Solve first-order ODEs or Systems of ODEs given initial conditions.")

    mode = st.radio("Solver Mode", ["Single Equation (y')", "System of Equations (Vector Field)"], horizontal=True)

    if mode == "Single Equation (y')":
        st.markdown("Solve $\\frac{dy}{dx} = f(x, y)$")
        
        # --- STIFF EQUATION PRESET ---
        with st.sidebar:
            st.header("Presets")
            if st.button("Load Stiff Equation Example"):
                st.session_state.ode_f_str = "-15*y"
                st.session_state.ode_y0 = 1.0
                st.session_state.ode_h = 0.15
                st.session_state.ode_x_end = 2.0
                st.session_state.ode_x0 = 0.0
                st.rerun()

        # --- 1. Configuration ---
        with st.container(border=True):
            c1, c2 = st.columns([2, 1])
            
            with c1:
                f_str = st.text_input("Enter f(x, y):", value="x + y", key="ode_f_str", help="Use Python syntax, e.g., 'x - y**2', 'sin(x)*y'")
                st.latex(f"\frac{{dy}}{{dx}} = {f_str}")
                exact_str = st.text_input("Exact Solution y(x) (optional):", value="", help="For error calculation. E.g. '2*exp(x) - x - 1'")
                
            with c2:
                methods = st.multiselect(
                    "Select Methods to Compare",
                    ["Euler", "Heun", "RK4", "RK45"],
                    default=["Euler", "RK4"]
                )
                
                if "RK45" in methods:
                    tol = st.number_input("Tolerance (RK45)", value=1e-4, format="%.1e", step=1e-5)

            c3, c4, c5, c6 = st.columns(4)
            with c3:
                x0 = st.number_input("x0 (Initial x)", value=0.0, step=0.1, key="ode_x0")
            with c4:
                y0 = st.number_input("y0 (Initial y)", value=1.0, step=0.1, key="ode_y0")
            with c5:
                h = st.number_input("Step Size (h)", value=0.1, min_value=0.001, step=0.01, format="%.3f", key="ode_h")
            with c6:
                x_end = st.number_input("End x", value=5.0, step=0.5, key="ode_x_end")

            solve_btn = st.button("Solve ODE", type="primary", icon=":material/play_arrow:")

        # --- 2. Calculation & Visualization ---
        if solve_btn:
            solver = ODESolver(f_str, x0, y0, h, x_end)
            
            if not solver.valid:
                st.error(f"Error parsing equation: {solver.error}")
                if "ode_results" in st.session_state:
                    del st.session_state.ode_results
            else:
                results = {}
                metrics = []
                
                # 0. Establish Ground Truth (Exact or High-Precision Numerical)
                y_truth_func = None
                if exact_str:
                    # Validate exact string
                    test_val = solver.evaluate_exact(exact_str, np.array([x0]))
                    if test_val is not None:
                        def y_truth_func(x): return solver.evaluate_exact(exact_str, x)
                else:
                    # Use high-precision RK45 as baseline if no exact solution provided
                    df_truth = solver.solve_rk45(tol=1e-9)
                    if df_truth is not None:
                        def y_truth_func(x):
                            return np.interp(x, df_truth['x'], df_truth['y_rk45'])

                # 1. Run Fixed Step Methods
                fixed_methods = [m for m in methods if m in ["Euler", "Heun", "RK4"]]
                
                for m in fixed_methods:
                    start_t = time.perf_counter()
                    if m == "Euler":
                        x_vals, y_vals = solver.solve_euler()
                    elif m == "Heun":
                        x_vals, y_vals = solver.solve_heun()
                    elif m == "RK4":
                        x_vals, y_vals = solver.solve_rk4()
                    end_t = time.perf_counter()
                    
                    if x_vals is not None:
                        results[m] = pd.DataFrame({'x': x_vals, 'y': y_vals})
                        
                        # Error Calculation
                        max_err = None
                        if y_truth_func:
                            try:
                                y_true = y_truth_func(x_vals)
                                if y_true is not None:
                                    max_err = np.nanmax(np.abs(y_vals - y_true))
                            except: pass
                        
                        metrics.append({
                            "Method": m,
                            "Time (s)": f"{end_t - start_t:.6f}",
                            "Max Error": f"{max_err:.6e}" if max_err is not None else "N/A"
                        })
                
                # 2. Run Adaptive Method
                if "RK45" in methods:
                    start_t = time.perf_counter()
                    df_rk45 = solver.solve_rk45(tol)
                    end_t = time.perf_counter()
                    
                    if df_rk45 is not None:
                        results["RK45"] = df_rk45.rename(columns={'y_rk45': 'y'})
                        
                        max_err = None
                        if y_truth_func:
                            try:
                                y_true = y_truth_func(df_rk45['x'].values)
                                if y_true is not None:
                                    max_err = np.nanmax(np.abs(df_rk45['y_rk45'].values - y_true))
                            except: pass
                                
                        metrics.append({
                            "Method": "RK45",
                            "Time (s)": f"{end_t - start_t:.6f}",
                            "Max Error": f"{max_err:.6e}" if max_err is not None else "N/A"
                        })
                
                st.session_state.ode_results = results
                st.session_state.ode_metrics = metrics
                st.session_state.ode_solver = solver

        if "ode_results" in st.session_state and st.session_state.ode_results:
            results = st.session_state.ode_results
            metrics = st.session_state.ode_metrics
            solver = st.session_state.ode_solver
            
            if results:
                # Render Graph
                render_ode_chart(results, solver.f)
                
                # Render Comparison Table
                if metrics:
                    st.subheader("Performance Comparison")
                    st.dataframe(pd.DataFrame(metrics), use_container_width=True)

                st.divider()
                st.subheader("Step-by-Step Solution")
                
                # Single selection section for detailed steps
                method_for_steps = st.selectbox("Select Method to View Steps", list(results.keys()))
                
                if method_for_steps:
                    render_ode_steps(solver, method_for_steps, results[method_for_steps])

            # --- CONVERGENCE ANALYSIS ---
            st.divider()
            with st.expander("📉 Convergence Analysis (Error vs Step Size)", expanded=False):
                st.markdown("Analyze how the Global Error decreases as we reduce the step size $h$.")
                if st.button("Analyze Error"):
                    h_vals = [0.1, 0.05, 0.025, 0.0125]
                    methods_to_test = ["Euler", "Heun", "RK4"]
                    error_data = {m: [] for m in methods_to_test}
                    
                    # Determine Truth
                    truth_solver = ODESolver(solver.f_str, solver.x0, solver.y0, 0.001, solver.x_end)
                    if exact_str:
                        # Use user exact string
                        def get_truth(x): return truth_solver.evaluate_exact(exact_str, x)
                    else:
                        # Use high-precision RK45
                        df_truth = truth_solver.solve_rk45(tol=1e-9)
                        if df_truth is None:
                            st.error("Could not generate reference solution.")
                            st.stop()
                        # Create interpolator for truth
                        def get_truth(x_query):
                            return np.interp(x_query, df_truth['x'], df_truth['y_rk45'])

                    for h_test in h_vals:
                        test_solver = ODESolver(solver.f_str, solver.x0, solver.y0, h_test, solver.x_end)
                        
                        # Euler
                        xe, ye = test_solver.solve_euler()
                        if xe is not None: error_data["Euler"].append(np.max(np.abs(ye - get_truth(xe))))
                        
                        # Heun
                        xh, yh = test_solver.solve_heun()
                        if xh is not None: error_data["Heun"].append(np.max(np.abs(yh - get_truth(xh))))
                        
                        # RK4
                        xr, yr = test_solver.solve_rk4()
                        if xr is not None: error_data["RK4"].append(np.max(np.abs(yr - get_truth(xr))))

                    # Plot Log-Log
                    fig_conv = go.Figure()
                    for m in methods_to_test:
                        if len(error_data[m]) == len(h_vals):
                            fig_conv.add_trace(go.Scatter(
                                x=h_vals, y=error_data[m], mode='lines+markers', name=m
                            ))
                    
                    fig_conv.update_layout(title="Global Error vs Step Size (Log-Log)", xaxis_type="log", yaxis_type="log", xaxis_title="Step Size (h)", yaxis_title="Max Global Error")
                    fig_conv.update_xaxes(autorange="reversed") # Smaller h on right usually, but standard is left-to-right small-to-large. Let's keep standard.
                    st.plotly_chart(fig_conv, use_container_width=True)

    else:
        # --- SYSTEM MODE ---
        st.markdown("Solve systems like Predator-Prey: $\\frac{dx}{dt} = \\alpha x - \\beta xy, \\quad \\frac{dy}{dt} = \\delta xy - \\gamma y$")
        
        if "sys_vars" not in st.session_state:
            st.session_state.sys_vars = [
                {"name": "x", "expr": "1.2*x - 0.6*x*y", "val": 2.0},
                {"name": "y", "expr": "-0.8*y + 0.3*x*y", "val": 1.0}
            ]

        with st.container(border=True):
            c_indep, c_add = st.columns([1, 3])
            with c_indep:
                indep_var = st.text_input("Independent Variable", value="t")
            
            st.markdown("#### Dependent Variables & Equations")
            
            # Dynamic Inputs
            to_remove = []
            for i, var in enumerate(st.session_state.sys_vars):
                c1, c2, c3, c4 = st.columns([1, 3, 1, 0.5])
                with c1:
                    var['name'] = st.text_input(f"Var {i+1}", value=var['name'], key=f"sys_name_{i}")
                with c2:
                    var['expr'] = st.text_input(f"d{var['name']}/d{indep_var} =", value=var['expr'], key=f"sys_expr_{i}")
                with c3:
                    var['val'] = st.number_input(f"{var['name']}(0)", value=var['val'], key=f"sys_val_{i}")
                with c4:
                    st.write("")
                    st.write("")
                    if st.button("X", key=f"rem_{i}"):
                        to_remove.append(i)
            
            if to_remove:
                for i in sorted(to_remove, reverse=True):
                    del st.session_state.sys_vars[i]
                st.rerun()

            if st.button("Add Variable", icon=":material/add:"):
                st.session_state.sys_vars.append({"name": "z", "expr": "", "val": 0.0})
                st.rerun()

            st.divider()
            
            c_params1, c_params2, c_params3 = st.columns(3)
            with c_params1:
                t0 = st.number_input(f"Start {indep_var}", value=0.0)
            with c_params2:
                t_end = st.number_input(f"End {indep_var}", value=20.0)
            with c_params3:
                h = st.number_input("Step Size (h)", value=0.05, format="%.3f")

            solve_sys_btn = st.button("Solve System", type="primary", icon=":material/hub:")

        if solve_sys_btn:
            func_strs = [v['expr'] for v in st.session_state.sys_vars]
            dep_vars = [v['name'] for v in st.session_state.sys_vars]
            y0s = [v['val'] for v in st.session_state.sys_vars]
            
            solver = SystemODESolver(func_strs, indep_var, dep_vars, t0, y0s, h, t_end)
            
            if not solver.valid:
                st.error(f"Error: {solver.error}")
            else:
                df = solver.solve()
                render_system_ode_chart(df, indep_var, dep_vars)
                
                with st.expander("Show Data"):
                    st.dataframe(df, use_container_width=True)
