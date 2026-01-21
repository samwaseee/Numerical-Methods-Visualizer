import streamlit as st
import numpy as np
import pandas as pd
from algorithms_linear import LinearSolver
from components.linear_results import render_direct_steps, render_iterative_graph


def show_linear_page():
    # Cleanup
    if st.button("Back to Home", icon=":material/home:"):
        st.session_state.page = "landing"
        st.query_params["page"] = "landing"
        st.rerun()

    st.title("System of Linear Equations Solver")
    st.markdown("Solve $Ax = B$ using direct or iterative methods.")

    # --- 1. CONFIGURATION ---
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            dim = st.number_input(
                "Dimension (N)", min_value=2, max_value=10, value=3, step=1
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
                "Matrix Inversion",
                "Jacobi Iteration",
                "Gauss-Seidel Iteration",
            ],
        )

    with c_params:
        if "Iteration" in method:
            max_iter = st.number_input("Max Iterations", 5, 1000, 50)
            tol = st.number_input("Tolerance", 1e-9, 1e-1, 1e-6, format="%.6f")

    if st.button(
        "Solve System",
        type="primary",
        use_container_width=True,
        icon=":material/calculate:",
    ):
        solver = LinearSolver(st.session_state.lin_A, st.session_state.lin_B)

        if method == "Gauss Elimination":
            x, steps, err = solver.gauss_elimination()
            if err:
                st.error(err)
            else:
                st.success(f"Solution: {x}")
                render_direct_steps(steps)

        elif method == "Gauss-Jordan Elimination":
            x, steps, err = solver.gauss_jordan()
            if err:
                st.error(err)
            else:
                st.success(f"Solution: {x}")
                render_direct_steps(steps)

        elif method == "Matrix Inversion":
            x, inv = solver.matrix_inversion()
            if x is None:
                st.error("Matrix is singular (not invertible).")
            else:
                st.success(f"Solution: {x}")
                with st.expander("Inverse Matrix"):
                    st.write(inv)

        elif method == "Jacobi Iteration":
            df = solver.jacobi_method(max_iter, tol)
            render_iterative_graph(df)

        elif method == "Gauss-Seidel Iteration":
            df = solver.gauss_seidel(max_iter, tol)
            render_iterative_graph(df)
