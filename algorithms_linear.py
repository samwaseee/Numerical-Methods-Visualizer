import numpy as np
import pandas as pd


class LinearSolver:
    def __init__(self, A, B):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.n = len(B)

    def _format_matrix(self, mat, vec):
        """Helper to format augmented matrix for display"""
        aug = np.column_stack((mat, vec))
        return aug.tolist()

    def attempt_dominance_reorder(self):
        """
        Attempts to reorder the matrix rows to achieve diagonal dominance using a greedy approach (partial pivoting).
        This is a heuristic and does not guarantee diagonal dominance, but it places larger elements on the diagonal,
        which often improves stability for iterative methods.
        Returns the reordered A, B, the permutation map, and a boolean indicating if dominance was achieved.
        """
        A = self.A.copy()
        B = self.B.copy()
        n = self.n
        permutation = np.arange(n)  # Tracks original row indices

        for i in range(n):
            # Find the row (from i to n-1) with the largest absolute value in column i
            pivot_row_relative = np.argmax(np.abs(A[i:, i]))
            pivot_row_absolute = i + pivot_row_relative

            if pivot_row_absolute != i:
                # Swap rows in the matrix and vector
                A[[i, pivot_row_absolute]], B[[i, pivot_row_absolute]] = A[[pivot_row_absolute, i]], B[[pivot_row_absolute, i]]

        is_dominant = np.all(np.abs(np.diag(A)) >= np.sum(np.abs(A), axis=1) - np.abs(np.diag(A)))
        return A, B, is_dominant

    def get_system_analysis(self):
        """Performs diagnostic checks on the linear system."""
        analysis = {}
        rows, cols = self.A.shape
        
        # 1. Determinant & Condition Number (Square matrices only)
        if rows == cols:
            try:
                det = np.linalg.det(self.A)
                cond = np.linalg.cond(self.A)
            except np.linalg.LinAlgError:
                det = 0.0
                cond = float('inf')
        else:
            det = None
            cond = None

        analysis['determinant'] = det
        analysis['condition_number'] = cond
        
        # 2. Ranks & Solvability
        rank_A = np.linalg.matrix_rank(self.A)
        aug = np.column_stack((self.A, self.B))
        rank_Aug = np.linalg.matrix_rank(aug)
        
        if rank_A < rank_Aug:
            status = "Inconsistent (No Solution)"
            desc = "Parallel lines/planes (Rank A < Rank A|B)"
        elif rank_A == rank_Aug:
            if rank_A == cols:
                status = "Consistent (Unique Solution)"
                desc = "Lines/Planes intersect at one point"
            else:
                status = "Consistent (Infinite Solutions)"
                desc = "Coincident lines/planes (Rank A < N)"
        else:
            status = "Unknown"
            desc = "-"
            
        analysis['rank_A'] = rank_A
        analysis['rank_Aug'] = rank_Aug
        analysis['solvability'] = status
        analysis['solvability_desc'] = desc
        
        # 3. Diagonal Dominance
        rows_status = []
        is_dominant = True
        for i in range(rows):
            diag = abs(self.A[i, i]) if i < cols else 0
            off_diag = np.sum(np.abs(self.A[i, :])) - diag
            row_is_dom = diag >= off_diag
            if not row_is_dom:
                is_dominant = False
            rows_status.append({
                "row": i+1,
                "diag": diag,
                "sum_off": off_diag,
                "is_dominant": row_is_dom
            })
            
        analysis['diagonal_dominance'] = {
            "is_dominant": is_dominant,
            "rows": rows_status
        }

        # 4. Attempt to reorder for dominance
        A_reordered, B_reordered, is_dominant_reordered = self.attempt_dominance_reorder()
        reorder_is_different = not np.array_equal(self.A, A_reordered)

        if is_dominant_reordered and not is_dominant:
            reorder_status = "Success: A diagonally dominant form was found by reordering rows."
            reorder_suggestion = True
        elif reorder_is_different:
            reorder_status = "Heuristic Applied: Rows were reordered to place larger elements on the diagonal. This may improve convergence even if dominance was not achieved."
            reorder_suggestion = True
        else:
            reorder_status = "No reordering was necessary or effective."
            reorder_suggestion = False

        analysis['reorder_for_dominance'] = {"A_reordered": A_reordered.tolist(), "B_reordered": B_reordered.tolist(), "status_message": reorder_status, "is_suggested": reorder_suggestion}
        
        # 5. Eigenvalues & Eigenvectors (New)
        if rows == cols:
            try:
                vals, vecs = np.linalg.eig(self.A)
                # Format for display: Complex numbers need careful handling
                eigen_data = []
                for i in range(len(vals)):
                    eigen_data.append({
                        "index": i+1,
                        "value": vals[i],
                        "vector": vecs[:, i].tolist() # The column is the vector
                    })
                analysis['eigen'] = eigen_data
            except:
                analysis['eigen'] = None
        else:
            analysis['eigen'] = None

        return analysis

    def calculate_spectral_radius(self, method):
        """Calculates the spectral radius of the iteration matrix."""
        try:
            if "Jacobi" in method:
                # T = I - D^-1 A
                if np.any(np.abs(np.diag(self.A)) < 1e-10): return None
                D_inv = np.diag(1.0 / np.diag(self.A))
                T = np.eye(self.n) - D_inv @ self.A
            elif "Gauss-Seidel" in method:
                # T = -(D+L)^-1 U
                M = np.tril(self.A) # D + L
                if np.linalg.det(M) == 0: return None
                U = np.triu(self.A, 1)
                M_inv = np.linalg.inv(M)
                T = -M_inv @ U
            else:
                return None
            
            eigenvalues = np.linalg.eigvals(T)
            return max(abs(eigenvalues))
        except:
            return None

    def gauss_elimination(self, pivoting=True):
        A = self.A.copy()
        B = self.B.copy()
        n = self.n
        steps = []

        # Initial State
        steps.append(
            {
                "description": "Initial Augmented Matrix",
                "matrix": self._format_matrix(A, B),
                "explanation": "Start with the augmented matrix $[A|B]$. Our goal is to convert $A$ into an Upper Triangular Matrix (zeros below diagonal)."
            }
        )

        # Forward Elimination
        for i in range(n):
            # Pivoting
            if pivoting:
                pivot_idx = i + np.argmax(np.abs(A[i:, i]))
                if pivot_idx != i:
                    A[[i, pivot_idx]] = A[[pivot_idx, i]]
                    B[[i, pivot_idx]] = B[[pivot_idx, i]]
                    steps.append(
                        {
                            "description": f"Swap Row {i+1} $\leftrightarrow$ Row {pivot_idx+1} (Pivoting)",
                            "matrix": self._format_matrix(A, B),
                            "explanation": f"**Why?** The element at $A_{{{i+1},{i+1}}}$ is small. We swap with Row {pivot_idx+1} to bring the largest absolute value to the pivot position. This prevents division by zero and reduces rounding errors."
                        }
                    )

            if abs(A[i, i]) < 1e-10:
                return None, steps, "Zero pivot encountered. Matrix may be singular."

            for j in range(i + 1, n):
                factor = A[j, i] / A[i, i]
                if abs(factor) > 1e-10:
                    target_val = A[j, i]
                    pivot_val = A[i, i]
                    
                    A[j, i:] -= factor * A[i, i:]
                    B[j] -= factor * B[i]
                    steps.append(
                        {
                            "description": f"$R_{{{j+1}}} \\leftarrow R_{{{j+1}}} - ({factor:.4f}) R_{{{i+1}}}$",
                            "matrix": self._format_matrix(A, B),
                            "explanation": f"**Goal:** Eliminate $x_{{{i+1}}}$ from Row {j+1}.\n**Multiplier Calculation:** We need to zero out $A_{{{j+1},{i+1}}}$ ({target_val:.4f}).\nMultiplier = $\\frac{{\\text{{Target}}}}{{\\text{{Pivot}}}} = \\frac{{{target_val:.4f}}}{{{pivot_val:.4f}}} = {factor:.4f}$.\n**Operation:** Subtract {factor:.4f} $\\times$ Row {i+1} from Row {j+1}."
                        }
                    )

        # Back Substitution
        x = np.zeros(n)
        
        steps.append({
            "description": "Phase 2: Back Substitution",
            "matrix": self._format_matrix(A, B),
            "explanation": "The matrix is now in Upper Triangular form. We solve for variables starting from the bottom row ($x_N$) upwards."
        })

        try:
            for i in range(n - 1, -1, -1):
                sum_ax = sum(A[i, j] * x[j] for j in range(i + 1, n))
                x[i] = (B[i] - sum_ax) / A[i, i]
                
                # Explanation for variable extraction
                rhs_terms = []
                for j in range(i+1, n):
                    rhs_terms.append(f"({A[i,j]:.4f} \\cdot {x[j]:.4f})")
                
                subtracted_part = " + ".join(rhs_terms) if rhs_terms else "0"
                
                steps.append({
                    "description": f"Solve for $x_{{{i+1}}}$",
                    "matrix": self._format_matrix(A, B),
                    "explanation": f"**Equation from Row {i+1}:**\n${A[i,i]:.4f} \\cdot x_{{{i+1}}} + {subtracted_part} = {B[i]:.4f}$\n\n**Solve:**\n$x_{{{i+1}}} = \\frac{{{B[i]:.4f} - ({sum_ax:.4f})}}{{{A[i,i]:.4f}}} = {x[i]:.4f}$"
                })
                
        except ZeroDivisionError:
            return None, steps, "Infinite solutions or no solution."

        return x, steps, None

    def gauss_jordan(self):
        A = self.A.copy()
        B = self.B.copy()
        n = self.n
        steps = []

        steps.append(
            {
                "description": "Initial Matrix", 
                "matrix": self._format_matrix(A, B),
                "explanation": "Start with the augmented matrix. Goal: Convert $A$ into the Identity Matrix (Diagonal with 1s)."
            }
        )

        for i in range(n):
            # Pivot
            pivot_idx = i + np.argmax(np.abs(A[i:, i]))
            if pivot_idx != i:
                A[[i, pivot_idx]] = A[[pivot_idx, i]]
                B[[i, pivot_idx]] = B[[pivot_idx, i]]
                steps.append(
                    {
                        "description": f"Swap $R_{{{i+1}}} \\leftrightarrow R_{{{pivot_idx+1}}}$",
                        "matrix": self._format_matrix(A, B),
                        "explanation": f"**Why?** Swap Row {i+1} with Row {pivot_idx+1} to get the largest pivot for stability."
                    }
                )

            # Normalize pivot row
            pivot_val = A[i, i]
            if abs(pivot_val) < 1e-10:
                return None, steps, "Singular Matrix"

            if abs(pivot_val - 1.0) > 1e-10:
                A[i] /= pivot_val
                B[i] /= pivot_val
                steps.append(
                    {
                        "description": f"$R_{{{i+1}}} \\leftarrow R_{{{i+1}}} / {pivot_val:.4f}$",
                        "matrix": self._format_matrix(A, B),
                        "explanation": f"**Goal:** Make the pivot 1.\n**How:** Divide the entire Row {i+1} by {pivot_val:.4f}."
                    }
                )

            # Eliminate all other rows
            for j in range(n):
                if i != j:
                    factor = A[j, i]
                    if abs(factor) > 1e-10:
                        target_val = A[j, i]
                        A[j] -= factor * A[i]
                        B[j] -= factor * B[i]
                        steps.append(
                            {
                                "description": f"$R_{{{j+1}}} \\leftarrow R_{{{j+1}}} - ({factor:.4f}) R_{{{i+1}}}$",
                                "matrix": self._format_matrix(A, B),
                                "explanation": f"**Goal:** Eliminate $x_{{{i+1}}}$ from Row {j+1}.\n**Multiplier Calculation:** Target is {target_val:.4f}. Pivot is 1.\nMultiplier = {target_val:.4f}.\n**Operation:** Subtract {factor:.4f} $\\times$ Row {i+1} from Row {j+1}."
                            }
                        )
        
        # Solution Extraction
        sol_str = "\n".join([f"$x_{{{k+1}}} = {B[k]:.4f}$" for k in range(n)])
        steps.append({
            "description": "Solution Extraction",
            "matrix": self._format_matrix(A, B),
            "explanation": f"The matrix is now in Reduced Row Echelon Form (Identity). The solution vector is the last column:\n{sol_str}"
        })

        return B, steps, None

    def matrix_inversion(self):
        n = self.n
        # We operate on copies to avoid modifying state
        M = self.A.copy()
        I = np.eye(n)
        
        steps = []
        steps.append({
            "description": "Initial Setup $[A | I]$",
            "matrix": self._format_matrix(M, I),
            "explanation": "We augment matrix $A$ with the Identity Matrix $I$. We will perform row operations to transform $A$ into $I$. The operations will transform $I$ into $A^{-1}$."
        })

        for i in range(n):
            # 1. Pivoting
            pivot_idx = i + np.argmax(np.abs(M[i:, i]))
            if abs(M[pivot_idx, i]) < 1e-10:
                return None, None, steps, "Matrix is Singular (Determinant is 0)."

            if pivot_idx != i:
                M[[i, pivot_idx]] = M[[pivot_idx, i]]
                I[[i, pivot_idx]] = I[[pivot_idx, i]]
                steps.append({
                    "description": f"Swap Row {i+1} $\\leftrightarrow$ Row {pivot_idx+1}",
                    "matrix": self._format_matrix(M, I),
                    "explanation": f"Pivoting: Swapping rows to get largest element ({M[i,i]:.4f}) on diagonal for stability."
                })

            # 2. Normalize Pivot
            pivot = M[i, i]
            if abs(pivot - 1.0) > 1e-10:
                M[i] /= pivot
                I[i] /= pivot
                steps.append({
                    "description": f"$R_{{{i+1}}} \\leftarrow R_{{{i+1}}} / {pivot:.4f}$",
                    "matrix": self._format_matrix(M, I),
                    "explanation": f"**Goal:** Make the pivot 1.\n**Operation:** Divide Row {i+1} by {pivot:.4f}."
                })

            # 3. Eliminate Column
            for j in range(n):
                if i != j:
                    factor = M[j, i]
                    if abs(factor) > 1e-10:
                        M[j] -= factor * M[i]
                        I[j] -= factor * I[i]
                        steps.append({
                            "description": f"$R_{{{j+1}}} \\leftarrow R_{{{j+1}}} - ({factor:.4f}) R_{{{i+1}}}$",
                            "matrix": self._format_matrix(M, I),
                            "explanation": f"**Goal:** Eliminate $x_{{{i+1}}}$ from Row {j+1}.\n**Multiplier:** Target is {factor:.4f}. Pivot is 1.\n**Operation:** Subtract {factor:.4f} $\\times$ Row {i+1} from Row {j+1}."
                        })

        inv_A = I
        x = inv_A @ self.B

        # Explanation for X = A^-1 * B
        calc_lines = []
        for i in range(n):
            row_terms = [f"({inv_A[i, j]:.4f} \\cdot {self.B[j]:.4f})" for j in range(n)]
            equation = " + ".join(row_terms)
            calc_lines.append(f"$x_{{{i+1}}} = {equation} = {x[i]:.4f}$")
            
        steps.append({
            "description": "Compute $X = A^{-1}B$",
            "matrix": self._format_matrix(inv_A, x),
            "explanation": "**Matrix Multiplication Step:**\nWe multiply the Inverse Matrix by the constant vector $B$:\n\n" + "\n\n".join(calc_lines)
        })
        
        return x, inv_A, steps, None

    def cramers_rule(self):
        A = self.A.copy()
        B = self.B.copy()
        n = self.n
        steps = []

        if A.shape[0] != A.shape[1]:
            return None, steps, "Cramer's Rule requires a square matrix."

        try:
            det_A = np.linalg.det(A)
        except np.linalg.LinAlgError:
            return None, steps, "Error calculating determinant."

        steps.append({
            "description": "Calculate Determinant of A",
            "matrix": A.tolist(),
            "explanation": f"First, we calculate the determinant of the coefficient matrix $D = |A|$.\n\n$D = {det_A:.4f}$"
        })

        if abs(det_A) < 1e-10:
            return None, steps, "Determinant is zero (Singular Matrix). Cramer's Rule cannot be applied."

        x = np.zeros(n)
        
        for i in range(n):
            A_i = A.copy()
            A_i[:, i] = B
            det_Ai = np.linalg.det(A_i)
            x[i] = det_Ai / det_A
            
            steps.append({
                "description": f"Calculate $D_{{{i+1}}}$ (Replace Column {i+1} with B)",
                "matrix": A_i.tolist(),
                "explanation": f"**Replace Column {i+1}:** We replace the {i+1}-th column of $A$ with the vector $B$.\n\n**Determinant:** $D_{{{i+1}}} = {det_Ai:.4f}$\n\n**Solve:** $x_{{{i+1}}} = \\frac{{D_{{{i+1}}}}}{{D}} = \\frac{{{det_Ai:.4f}}}{{{det_A:.4f}}} = {x[i]:.4f}$"
            })
            
        return x, steps, None

    def lu_decomposition(self):
        """
        Implements Doolittle's Method (A = LU).
        Restrictions: Requires non-zero diagonal pivots (no pivoting implemented for simplicity).
        """
        n = self.n
        A = self.A.copy()  # This will become U
        L = np.eye(n)      # This will become L (start as Identity)
        B = self.B.copy()
        steps = []

        steps.append({
            "description": "Initialization",
            "type": "decomposition",
            "L": L.tolist(),
            "U": A.tolist(),
            "explanation": "Start with $L = I$ (Identity) and $U = A$. We will perform elimination on $U$ and store the multipliers in $L$."
        })

        # --- Phase 1: Decomposition (A = LU) ---
        for i in range(n):
            # Pivot Check
            if abs(A[i, i]) < 1e-10:
                return None, None, None, steps, "Zero pivot encountered. Simple LU Decomposition requires non-zero diagonal elements. Try reordering rows manually."

            for j in range(i + 1, n):
                factor = A[j, i] / A[i, i]
                
                # Store multiplier in L
                L[j, i] = factor
                
                # Update U (Eliminate Row j)
                original_row = A[j].copy()
                A[j, i:] -= factor * A[i, i:]
                
                steps.append({
                    "description": f"Eliminate $A_{{{j+1},{i+1}}}$",
                    "type": "decomposition",
                    "L": L.tolist(),
                    "U": A.tolist(),
                    "explanation": f"**Multiplier:** $L_{{{j+1},{i+1}}} = \\frac{{{original_row[i]:.4f}}}{{{A[i,i]:.4f}}} = {factor:.4f}$\n\n**Update U:** Subtract {factor:.4f} $\\times$ Row {i+1} from Row {j+1}."
                })

        # --- Phase 2: Forward Substitution (Ly = B) ---
        y = np.zeros(n)
        steps.append({
            "description": "Start Forward Substitution ($Ly = B$)",
            "type": "solve",
            "matrix": self._format_matrix(L, B),
            "explanation": "We first solve for the intermediate vector $y$ using the Lower Triangular matrix $L$."
        })
        
        for i in range(n):
            s = sum(L[i, j] * y[j] for j in range(i))
            y[i] = B[i] - s
        
        # --- Phase 3: Backward Substitution (Ux = y) ---
        x = np.zeros(n)
        steps.append({
            "description": "Start Backward Substitution ($Ux = y$)",
            "type": "solve",
            "matrix": self._format_matrix(A, y), # A is now U
            "explanation": "Now we use the Upper Triangular matrix $U$ and the vector $y$ to solve for $x$."
        })

        for i in range(n - 1, -1, -1):
            s = sum(A[i, j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - s) / A[i, i]

        return x, L, A, steps, None

    def cholesky_decomposition(self):
        """
        Implements Cholesky Decomposition (A = L * L.T).
        Requirement: Matrix must be Symmetric and Positive Definite.
        """
        n = self.n
        A = self.A.copy()
        
        # 1. Symmetry Check
        if not np.allclose(A, A.T, atol=1e-8):
            return None, None, [], "Matrix is not Symmetric. Cholesky requires A = A^T."

        L = np.zeros_like(A)
        steps = []
        
        steps.append({
            "description": "Initialization",
            "type": "decomposition",
            "L": np.zeros_like(A).tolist(), # Empty L
            "U": A.tolist(), # Show A
            "explanation": "Goal: Decompose $A$ into $L L^T$. We iterate through rows and columns to calculate $L_{ij}$."
        })

        try:
            for i in range(n):
                for j in range(i + 1):
                    sum_k = sum(L[i, k] * L[j, k] for k in range(j))
                    
                    if i == j:
                        # Diagonal Element: L_ii = sqrt(A_ii - sum(L_ik^2))
                        val = A[i, i] - sum_k
                        if val <= 0:
                            return None, None, steps, "Matrix is not Positive Definite (found non-positive diagonal during decompostion)."
                        L[i, j] = np.sqrt(val)
                        
                        steps.append({
                            "description": f"Calculate Diagonal $L_{{{i+1},{i+1}}}$",
                            "type": "decomposition",
                            "L": L.tolist(),
                            "U": L.T.tolist(), # Show L^T as U
                            "explanation": f"$L_{{{i+1},{i+1}}} = \\sqrt{{A_{{{i+1},{i+1}}} - \\sum L_{{ik}}^2}} = \\sqrt{{{val:.4f}}} = {L[i,j]:.4f}$"
                        })
                    else:
                        # Off-Diagonal: L_ij = (1/L_jj) * (A_ij - sum(L_ik * L_jk))
                        L[i, j] = (1.0 / L[j, j]) * (A[i, j] - sum_k)
                        
                        steps.append({
                            "description": f"Calculate $L_{{{i+1},{j+1}}}$",
                            "type": "decomposition",
                            "L": L.tolist(),
                            "U": L.T.tolist(),
                            "explanation": f"$L_{{{i+1},{j+1}}} = \\frac{{1}}{{L_{{{j+1},{j+1}}}}} (A_{{{i+1},{j+1}}} - \\sum L_{{ik}}L_{{jk}}) = {L[i,j]:.4f}$"
                        })
                        
        except Exception as e:
            return None, None, steps, f"Error during decomposition: {str(e)}"

        # Solve L y = B
        y = np.zeros(n)
        for i in range(n):
            s = sum(L[i, j] * y[j] for j in range(i))
            y[i] = (self.B[i] - s) / L[i, i]
            
        # Solve L.T x = y
        x = np.zeros(n)
        LT = L.T
        for i in range(n - 1, -1, -1):
            s = sum(LT[i, j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - s) / LT[i, i]

        return x, L, steps, None

    def jacobi_method(self, initial_guess=None, max_iter=50, tol=1e-6):
        if initial_guess is None:
            x = np.zeros(self.n)
        else:
            x = np.array(initial_guess, dtype=float)
            if x.shape != (self.n,):
                return pd.DataFrame([{"Error": "Initial guess has the wrong dimension."}])

        history = []

        # Check diagonal dominance (warning only)
        diag = np.abs(np.diag(self.A))
        off_diag = np.sum(np.abs(self.A), axis=1) - diag
        if np.any(diag <= off_diag):
            # Just a note in the first row
            pass

        for k in range(max_iter):
            x_new = np.zeros_like(x)
            for i in range(self.n):
                s = sum(self.A[i, j] * x[j] for j in range(self.n) if j != i)
                if abs(self.A[i, i]) < 1e-12:
                    return pd.DataFrame([{"Error": f"Division by zero on row {i+1}. Matrix may be singular."}])
                x_new[i] = (self.B[i] - s) / self.A[i, i]

            err = np.linalg.norm(x_new - x, ord=np.inf)
            row = {"Iter": k + 1, "Error": err}
            for i in range(self.n):
                row[f"x{i+1}"] = x_new[i]
            history.append(row)

            x = x_new
            if err < tol:
                break

        return pd.DataFrame(history)

    def gauss_seidel(self, initial_guess=None, max_iter=50, tol=1e-6):
        if initial_guess is None:
            x = np.zeros(self.n)
        else:
            x = np.array(initial_guess, dtype=float)
            if x.shape != (self.n,):
                return pd.DataFrame([{"Error": "Initial guess has the wrong dimension."}])

        history = []

        for k in range(max_iter):
            x_old = x.copy()
            for i in range(self.n):
                s1 = sum(self.A[i, j] * x[j] for j in range(i))
                s2 = sum(self.A[i, j] * x_old[j] for j in range(i + 1, self.n))
                if abs(self.A[i, i]) < 1e-12:
                    return pd.DataFrame([{"Error": f"Division by zero on row {i+1}. Matrix may be singular."}])
                x[i] = (self.B[i] - s1 - s2) / self.A[i, i]

            err = np.linalg.norm(x - x_old, ord=np.inf)
            row = {"Iter": k + 1, "Error": err}
            for i in range(self.n):
                row[f"x{i+1}"] = x[i]
            history.append(row)

            if err < tol:
                break

        return pd.DataFrame(history)
