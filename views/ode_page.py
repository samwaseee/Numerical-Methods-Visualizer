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
        
        # --- 1. Configuration ---
        with st.container(border=True):
            # --- Example Selector ---
            examples = {
                "Custom": {"f": "x + y", "exact": "", "x0": 0.0, "y0": 1.0, "h": 0.1, "x_end": 5.0},
                "Linear (y' = x + y)": {"f": "x + y", "exact": "2*exp(x) - x - 1", "x0": 0.0, "y0": 1.0, "h": 0.1, "x_end": 2.0},
                "Exponential Growth (y' = y)": {"f": "y", "exact": "exp(x)", "x0": 0.0, "y0": 1.0, "h": 0.1, "x_end": 2.0},
                "Exponential Decay (y' = -y)": {"f": "-y", "exact": "10*exp(-x)", "x0": 0.0, "y0": 10.0, "h": 0.1, "x_end": 5.0},
                "Sine Wave (y' = cos(x))": {"f": "cos(x)", "exact": "sin(x)", "x0": 0.0, "y0": 0.0, "h": 0.1, "x_end": 6.3},
                "Logistic Growth (y' = y*(1-y))": {"f": "y*(1-y)", "exact": "1 / (1 + exp(-x))", "x0": 0.0, "y0": 0.5, "h": 0.1, "x_end": 5.0}
            }

            def update_ode_inputs():
                sel = st.session_state.ode_example_sel
                if sel in examples and sel != "Custom":
                    data = examples[sel]
                    st.session_state.ode_f_str = data["f"]
                    st.session_state.ode_exact_str = data["exact"]
                    st.session_state.ode_x0 = data["x0"]
                    st.session_state.ode_y0 = data["y0"]
                    st.session_state.ode_h = data["h"]
                    st.session_state.ode_x_end = data["x_end"]

            c1, c2 = st.columns([2, 1])
            
            with c1:
                f_str = st.text_input("Enter f(x, y):", value="x + y", key="ode_f_str", help="Use Python syntax, e.g., 'x - y**2', 'sin(x)*y'")
                st.latex(f"\frac{{dy}}{{dx}} = {f_str}")
                exact_str = st.text_input("Exact Solution y(x) (optional):", value="", key="ode_exact_str", help="For error calculation. E.g. '2*exp(x) - x - 1'")
                
            with c2:
                st.selectbox("Select Example", list(examples.keys()), key="ode_example_sel", on_change=update_ode_inputs)
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
            with st.expander("📉 Convergence Analysis (Error vs Step Size)", expanded=True):
                st.markdown("Analyze how the Global Error decreases as we reduce the step size $h$.")
                
                c_h1, c_h2 = st.columns(2)
                with c_h1:
                    h_start = st.number_input("Starting Step Size (h)", value=0.1, format="%.4f", key="conv_h_start")
                with c_h2:
                    n_steps = st.number_input("Number of Halving Steps", value=5, min_value=3, max_value=10, key="conv_n_steps")

                if st.button("Analyze Error"):
                    h_vals = [h_start / (2**i) for i in range(n_steps)]
                    
                    # Filter selected methods to only include fixed-step ones for this analysis
                    methods_to_test = [m for m in methods if m in ["Euler", "Heun", "RK4"]]
                    
                    if not methods_to_test:
                        st.warning("Please select at least one fixed-step method (Euler, Heun, or RK4) to run the analysis.")
                        st.stop()

                    error_data = {m: [] for m in methods_to_test}
                    
                    with st.spinner("Analyzing convergence..."):
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
                            
                            if "Euler" in methods_to_test:
                                xe, ye = test_solver.solve_euler()
                                if xe is not None: error_data["Euler"].append(np.max(np.abs(ye - get_truth(xe))))
                            
                            if "Heun" in methods_to_test:
                                xh, yh = test_solver.solve_heun()
                                if xh is not None: error_data["Heun"].append(np.max(np.abs(yh - get_truth(xh))))
                            
                            if "RK4" in methods_to_test:
                                xr, yr = test_solver.solve_rk4()
                                if xr is not None: error_data["RK4"].append(np.max(np.abs(yr - get_truth(xr))))

                    # Plot Log-Log
                    fig_conv = go.Figure()
                    for m in methods_to_test:
                        if len(error_data.get(m, [])) == len(h_vals):
                            fig_conv.add_trace(go.Scatter(
                                x=h_vals, y=error_data[m], mode='lines+markers', name=m
                            ))
                    
                    fig_conv.update_layout(title="Global Error vs Step Size (Log-Log)", xaxis_type="log", yaxis_type="log", xaxis_title="Step Size (h)", yaxis_title="Max Global Error")
                    st.plotly_chart(fig_conv, use_container_width=True)

    else:
        # --- SYSTEM MODE ---
        with st.expander("📘 Theory: Systems of ODEs", expanded=False):
            st.markdown(r"""
            **Systems of First-Order ODEs** involve multiple dependent variables changing with respect to a single independent variable $t$.
            
            $$ \frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y}) $$
            
            where $\mathbf{y} = [y_1, y_2, \dots, y_n]^T$.
            
            Numerical methods like **RK4** generalize naturally to systems by treating variables as a vector.
            
            *   **Coupled Dynamics:** A change in one variable ($y_1$) affects the rate of change of others ($y_2$).
            *   **Phase Space:** We often visualize solutions as trajectories in the phase plane ($y_1$ vs $y_2$) rather than just time series.
            *   **Error:** Global error scales with step size $h$ similar to single equations ($O(h^4)$ for RK4).
            """)

        st.markdown("Solve systems like Predator-Prey: $\\frac{dx}{dt} = \\alpha x - \\beta xy, \\quad \\frac{dy}{dt} = \\delta xy - \\gamma y$")
        
        if "sys_vars" not in st.session_state:
            st.session_state.sys_vars = [
                {"name": "x", "expr": "1.2*x - 0.6*x*y", "val": 2.0, "exact": ""},
                {"name": "y", "expr": "-0.8*y + 0.3*x*y", "val": 1.0, "exact": ""}
            ]

        with st.container(border=True):
            c_indep, c_add = st.columns([1, 3])
            with c_indep:
                indep_var = st.text_input("Independent Variable", value="t")
            
            st.markdown("#### Dependent Variables & Equations")
            
            # Dynamic Inputs
            to_remove = []
            for i, var in enumerate(st.session_state.sys_vars):
                if 'exact' not in var: var['exact'] = ""

                c1, c2, c3, c4 = st.columns([1, 3, 1, 0.5])
                with c1:
                    var['name'] = st.text_input(f"Var {i+1}", value=var['name'], key=f"sys_name_{i}")
                with c2:
                    var['expr'] = st.text_input(f"d{var['name']}/d{indep_var} =", value=var['expr'], key=f"sys_expr_{i}")
                    var['exact'] = st.text_input(f"Exact {var['name']}({indep_var}) [Optional]", value=var['exact'], key=f"sys_exact_{i}", placeholder="e.g. 2*exp(t)")
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
                st.session_state.sys_vars.append({"name": "z", "expr": "", "val": 0.0, "exact": ""})
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
            exact_strs = [v.get('exact', '') for v in st.session_state.sys_vars]
            
            solver = SystemODESolver(func_strs, indep_var, dep_vars, t0, y0s, h, t_end)
            
            if not solver.valid:
                st.error(f"Error: {solver.error}")
                if "sys_ode_results" in st.session_state:
                    del st.session_state.sys_ode_results
            else:
                start_t = time.perf_counter()
                df = solver.solve()
                end_t = time.perf_counter()
                
                # --- Metrics & Error Calculation ---
                metrics = [{"Metric": "Execution Time", "Value": f"{end_t - start_t:.6f} s"}]
                
                safe_dict = {k: getattr(np, k) for k in dir(np) if not k.startswith('_')}
                safe_dict[indep_var] = df[indep_var].values
                
                has_exact = False
                for i, var_name in enumerate(dep_vars):
                    exact_expr = exact_strs[i]
                    if exact_expr.strip():
                        try:
                            y_true = eval(exact_expr, {"__builtins__": {}}, safe_dict)
                            error = np.abs(df[var_name].values - y_true)
                            max_err = np.max(error)
                            metrics.append({"Metric": f"Max Error ({var_name})", "Value": f"{max_err:.6e}"})
                            has_exact = True
                        except Exception:
                            metrics.append({"Metric": f"Max Error ({var_name})", "Value": "Eval Error"})

                st.session_state.sys_ode_results = {
                    "df": df,
                    "metrics": metrics,
                    "indep_var": indep_var,
                    "dep_vars": dep_vars,
                    "has_exact": has_exact,
                    "func_strs": func_strs,
                    "t0": t0,
                    "y0s": y0s,
                    "h": h,
                    "t_end": t_end,
                    "exact_strs": exact_strs
                }

        if "sys_ode_results" in st.session_state:
            res = st.session_state.sys_ode_results
            render_system_ode_chart(res["df"], res["indep_var"], res["dep_vars"])
            
            st.subheader("Performance & Accuracy")
            st.dataframe(pd.DataFrame(res["metrics"]), use_container_width=True, hide_index=True)
            
            with st.expander("Show Data"):
                st.dataframe(res["df"], use_container_width=True)

            # --- CONVERGENCE ANALYSIS (SYSTEMS) ---
            st.divider()
            with st.expander("📉 Convergence Analysis (System)", expanded=False):
                st.markdown("Analyze how the error decreases as we reduce step size $h$.")
                c_h1, c_h2 = st.columns(2)
                with c_h1: h_start_sys = st.number_input("Starting h", value=res["h"], format="%.4f", key="sys_conv_h")
                with c_h2: n_steps_sys = st.number_input("Halving Steps", value=4, min_value=3, max_value=8, key="sys_conv_n")
                
                if st.button("Analyze System Convergence"):
                    if not res["has_exact"]:
                        st.warning("Exact solutions required for convergence analysis.")
                    else:
                        h_vals = [h_start_sys / (2**k) for k in range(n_steps_sys)]
                        max_errors = []
                        with st.spinner("Running convergence test..."):
                            for h_test in h_vals:
                                test_solver = SystemODESolver(res["func_strs"], res["indep_var"], res["dep_vars"], res["t0"], res["y0s"], h_test, res["t_end"])
                                df_test = test_solver.solve()
                                safe_dict_test = {k: getattr(np, k) for k in dir(np) if not k.startswith('_')}
                                safe_dict_test[res["indep_var"]] = df_test[res["indep_var"]].values
                                
                                current_max_err = 0.0
                                for i, var_name in enumerate(res["dep_vars"]):
                                    if res["exact_strs"][i].strip():
                                        try:
                                            y_true_test = eval(res["exact_strs"][i], {"__builtins__": {}}, safe_dict_test)
                                            err = np.max(np.abs(df_test[var_name].values - y_true_test))
                                            if err > current_max_err: current_max_err = err
                                        except: pass
                                max_errors.append(current_max_err)
                        
                        fig_conv = go.Figure()
                        fig_conv.add_trace(go.Scatter(x=h_vals, y=max_errors, mode='lines+markers', name='System Max Error'))
                        fig_conv.update_layout(title="System Global Error vs Step Size (Log-Log)", xaxis_type="log", yaxis_type="log", xaxis_title="Step Size (h)", yaxis_title="Max Error")
                        st.plotly_chart(fig_conv, use_container_width=True)
