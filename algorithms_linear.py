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
                        }
                    )

            if abs(A[i, i]) < 1e-10:
                return None, steps, "Zero pivot encountered. Matrix may be singular."

            for j in range(i + 1, n):
                factor = A[j, i] / A[i, i]
                if abs(factor) > 1e-10:
                    A[j, i:] -= factor * A[i, i:]
                    B[j] -= factor * B[i]
                    steps.append(
                        {
                            "description": f"$R_{{{j+1}}} \\leftarrow R_{{{j+1}}} - ({factor:.4f}) R_{{{i+1}}}$",
                            "matrix": self._format_matrix(A, B),
                        }
                    )

        # Back Substitution
        x = np.zeros(n)
        try:
            for i in range(n - 1, -1, -1):
                sum_ax = sum(A[i, j] * x[j] for j in range(i + 1, n))
                x[i] = (B[i] - sum_ax) / A[i, i]
        except ZeroDivisionError:
            return None, steps, "Infinite solutions or no solution."

        return x, steps, None

    def gauss_jordan(self):
        A = self.A.copy()
        B = self.B.copy()
        n = self.n
        steps = []

        steps.append(
            {"description": "Initial Matrix", "matrix": self._format_matrix(A, B)}
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
                    }
                )

            # Eliminate all other rows
            for j in range(n):
                if i != j:
                    factor = A[j, i]
                    if abs(factor) > 1e-10:
                        A[j] -= factor * A[i]
                        B[j] -= factor * B[i]
                        steps.append(
                            {
                                "description": f"$R_{{{j+1}}} \\leftarrow R_{{{j+1}}} - ({factor:.4f}) R_{{{i+1}}}$",
                                "matrix": self._format_matrix(A, B),
                            }
                        )

        return B, steps, None

    def matrix_inversion(self):
        try:
            inv_A = np.linalg.inv(self.A)
            x = inv_A @ self.B
            return x, inv_A
        except np.linalg.LinAlgError:
            return None, None

    def jacobi_method(self, max_iter=50, tol=1e-6):
        x = np.zeros(self.n)
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
                    return pd.DataFrame(history)  # Stop if div by zero
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

    def gauss_seidel(self, max_iter=50, tol=1e-6):
        x = np.zeros(self.n)
        history = []

        for k in range(max_iter):
            x_old = x.copy()
            for i in range(self.n):
                s1 = sum(self.A[i, j] * x[j] for j in range(i))
                s2 = sum(self.A[i, j] * x_old[j] for j in range(i + 1, self.n))
                if abs(self.A[i, i]) < 1e-12:
                    return pd.DataFrame(history)
                x[i] = (self.B[i] - s1 - s2) / self.A[i, i]

            err = np.linalg.norm(x - x_old, ord=np.inf)
            row = {"Iter": k + 1, "Error": err}
            for i in range(self.n):
                row[f"x{i+1}"] = x[i]
            history.append(row)

            if err < tol:
                break

        return pd.DataFrame(history)
