import streamlit as st
import numpy as np
import pandas as pd
from algorithms_linear import LinearSolver
from components.linear_results import render_direct_steps, render_iterative_graph, render_system_analysis
from components.linear_graph import render_linear_system_graph


def render_lu_steps(steps):
    st.subheader("LU Decomposition Steps")
    
    for i, step in enumerate(steps):
        with st.expander(f"Step {i}: {step['description']}", expanded=(i==0 or i==len(steps)-1)):
            if "explanation" in step:
                st.info(step["explanation"])

            if step.get("type") == "decomposition":
                # Render L and U side by side using LaTeX
                L_mat = np.array(step["L"])
                U_mat = np.array(step["U"])
                
                # Helper to make latex string
                def mat_to_tex(M):
                    lines = []
                    for row in M:
                        lines.append(" & ".join([f"{x:.4f}" for x in row]))
                    return "\\begin{bmatrix}" + "\\\\".join(lines) + "\\end{bmatrix}"

                st.latex(f"L = {mat_to_tex(L_mat)} \\quad U = {mat_to_tex(U_mat)}")
            
            elif step.get("type") == "solve":
                # Render standard augmented matrix for substitution phases
                mat = np.array(step['matrix'])
                rows_latex = []
                for row in mat:
                    coeffs = " & ".join([f"{x:.4f}" for x in row[:-1]])
                    const = f"{row[-1]:.4f}"
                    rows_latex.append(f"{coeffs} & {const}")
                
                latex_aug = f"\\left[\\begin{{array}}{{{'c'*len(mat) + '|c'}}}\n"
                latex_aug += " \\\\ \n".join(rows_latex)
                latex_aug += "\n\\end{array}\\right]"
                st.latex(latex_aug)

def show_linear_page():
    # Cleanup
    if st.button("Back to Home", icon=":material/home:"):
        for key in list(st.session_state.keys()):
            if key != 'dark_mode':
                del st.session_state[key]
        st.query_params.clear()
        st.session_state.page = "landing"
        st.query_params["page"] = "landing"
        st.rerun()

    st.title("System of Linear Equations Solver")
    st.markdown("Solve $Ax = B$ using direct or iterative methods.")

    # --- 1. CONFIGURATION ---
    with st.container(border=True):
        dim = st.number_input(
            "Dimension (N)", min_value=2, max_value=10, value=3, step=1, key="dimension"
        )

        # Initialize session state for matrix if dimension changes or not exists
        if "lin_A" not in st.session_state or st.session_state.lin_A.shape != (
            dim,
            dim,
        ):
            st.session_state.lin_A = np.random.randint(-10, 10, size=(dim, dim)).astype(float)
            st.session_state.lin_B = np.random.randint(-10, 10, size=dim).astype(float)
            
            # Clear widget state to force reload from new matrix values
            for key in list(st.session_state.keys()):
                if key.startswith("A_") or key.startswith("B_"):
                    del st.session_state[key]

        st.markdown("### Input Matrix A and Vector B")

        # Grid Layout: N columns for A, 1 spacer, 1 column for B
        cols = st.columns(dim + 2)

        # Header
        for j in range(dim):
            cols[j].markdown(
                f"<div style='text-align:center'><b>x {j+1}</b></div>",
                unsafe_allow_html=True,
            )
        cols[dim].markdown("")  # Spacer
        cols[dim + 1].markdown(
            "<div style='text-align:center'><b>= B</b></div>", unsafe_allow_html=True
        )

        # Rows
        for i in range(dim):
            cols = st.columns(dim + 2)
            for j in range(dim):
                val = cols[j].number_input(
                    f"A[{i},{j}]",
                    value=float(st.session_state.lin_A[i, j]),
                    key=f"A_{i}_{j}",
                    label_visibility="collapsed",
                )
                st.session_state.lin_A[i, j] = val

            cols[dim].markdown(
                "<div style='text-align:center; padding-top: 5px'>=</div>",
                unsafe_allow_html=True,
            )

            val_b = cols[dim + 1].number_input(
                f"B[{i}]",
                value=float(st.session_state.lin_B[i]),
                key=f"B_{i}",
                label_visibility="collapsed",
            )
            st.session_state.lin_B[i] = val_b

    # --- 2. METHOD SELECTION ---
    st.divider()
    c_method, c_params = st.columns([1, 1])

    with c_method:
        method = st.selectbox(
            "Method",
            [
                "Gauss Elimination",
                "Gauss-Jordan Elimination",
                "LU Decomposition (Doolittle)",
                "Cholesky Decomposition",
                "Matrix Inversion",
                "Cramer's Rule",
                "Jacobi Iteration",
                "Gauss-Seidel Iteration",
            ],
        )

    with c_params:
        if "Iteration" in method:
            max_iter = st.number_input("Max Iterations", 5, 1000, 50)
            tol = st.number_input("Tolerance", 1e-9, 1e-1, 1e-4, format="%.4f")

    if "Iteration" in method:
        with st.expander("Set Initial Guess (x₀)"):
            st.caption("Define initial values for the iteration (default is 0).")
            # Initialize session state for x0 if needed
            if "lin_x0" not in st.session_state or len(st.session_state.lin_x0) != dim:
                st.session_state.lin_x0 = np.zeros(dim)

            use_single_x0 = st.checkbox("Use same value for all variables", value=False, help="Sets x₀[1]...x₀[N] to the same number.")
            
            if use_single_x0:
                val = st.number_input("x₀ (all)", value=float(st.session_state.lin_x0[0]), key="x0_single")
                st.session_state.lin_x0 = np.full(dim, val)
            else:
                cols_x0 = st.columns(dim)
                for i in range(dim):
                    val_x0 = cols_x0[i].number_input(
                        f"x₀[{i+1}]",
                        value=float(st.session_state.lin_x0[i]),
                        key=f"x0_{i}",
                    )
                    st.session_state.lin_x0[i] = val_x0

    if st.button(
        "Solve System",
        type="primary",
        use_container_width=True,
        icon=":material/calculate:",
    ):
        # --- CALCULATION PHASE ---
        solver = LinearSolver(st.session_state.lin_A, st.session_state.lin_B)
        analysis = solver.get_system_analysis()
        
        result_data = {
            "method": method,
            "analysis": analysis,
            "solver_A": st.session_state.lin_A, # Store snapshot of input
            "solver_B": st.session_state.lin_B
        }

        if method == "Gauss Elimination":
            x, steps, err = solver.gauss_elimination()
            result_data.update({"x": x, "steps": steps, "err": err, "type": "direct"})

        elif method == "Gauss-Jordan Elimination":
            x, steps, err = solver.gauss_jordan()
            result_data.update({"x": x, "steps": steps, "err": err, "type": "direct"})

        elif method == "LU Decomposition (Doolittle)":
            x, L, U, steps, err = solver.lu_decomposition()
            result_data.update({"x": x, "L": L, "U": U, "steps": steps, "err": err, "type": "lu"})

        elif method == "Cholesky Decomposition":
            x, L, steps, err = solver.cholesky_decomposition()
            result_data.update({"x": x, "L": L, "U": L.T if L is not None else None, "steps": steps, "err": err, "type": "lu"})

        elif method == "Matrix Inversion":
            x, inv, steps, err = solver.matrix_inversion()
            result_data.update({"x": x, "inv": inv, "steps": steps, "err": err, "type": "inversion"})

        elif method == "Cramer's Rule":
            x, steps, err = solver.cramers_rule()
            result_data.update({"x": x, "steps": steps, "err": err, "type": "direct"})

        elif "Iteration" in method:
            reorder_info = analysis.get('reorder_for_dominance', {})
            use_reordered = reorder_info.get('is_suggested', False)
            
            iter_solver = solver
            if use_reordered:
                A_eff = np.array(reorder_info['A_reordered'])
                B_eff = np.array(reorder_info['B_reordered'])
                iter_solver = LinearSolver(A_eff, B_eff)
            
            initial_guess = st.session_state.get('lin_x0', np.zeros(solver.n))
            rho = solver.calculate_spectral_radius(method)
            
            if method == "Jacobi Iteration":
                df = iter_solver.jacobi_method(initial_guess, max_iter, tol)
            else:
                df = iter_solver.gauss_seidel(initial_guess, max_iter, tol)
                
            result_data.update({"df": df, "type": "iterative", "reorder_info": reorder_info, "spectral_radius": rho})

        st.session_state.lin_results = result_data

    # --- RENDERING PHASE (PERSISTENT) ---
    if "lin_results" in st.session_state:
        res = st.session_state.lin_results
        render_system_analysis(res['analysis'])

        if res['type'] in ["direct", "inversion"]:
            if res['err']:
                st.error(res['err'])
            else:
                st.success(f"Solution: {res['x']}")
                if res['type'] == "inversion":
                    with st.expander("Inverse Matrix"):
                        st.write(res['inv'])
                render_direct_steps(res['steps'])
                render_linear_system_graph(res['solver_A'], res['solver_B'], res['x'])

        elif res['type'] == "lu":
            if res['err']:
                st.error(res['err'])
            else:
                st.success(f"Solution: {res['x']}")
                render_lu_steps(res['steps'])
                render_linear_system_graph(res['solver_A'], res['solver_B'], res['x'])

        elif res['type'] == "iterative":
            reorder_info = res.get('reorder_info', {})
            if reorder_info.get('is_suggested', False):
                st.info(f"**Reordering Applied:** {reorder_info['status_message']}")
            
            df = res['df']
            rho = res.get('spectral_radius')
            render_iterative_graph(df, rho)
            if not df.empty and "Error" not in df.columns:
                x_cols = sorted([c for c in df.columns if c.startswith('x')])
                if x_cols:
                    sol = df.iloc[-1][x_cols].values
                    render_linear_system_graph(res['solver_A'], res['solver_B'], sol)
