import sympy as sp
import numpy as np
import pandas as pd
import re
from algorithms_interpolation import InterpolationSolver

class DifferentiationSolver:
    def __init__(self, x_data, y_data):
        self.interp = InterpolationSolver(x_data, y_data)
        self.x_sym = self.interp.x_sym

    def _evaluate_derivatives(self, expr, value):
        """
        Computes f(value), f'(value), f''(value) from the symbolic expression.
        """
        if expr is None:
            return None, "No expression generated."

        # Calculate derivatives symbolically
        d1_expr = sp.diff(expr, self.x_sym)
        d2_expr = sp.diff(d1_expr, self.x_sym)

        # Create callable functions
        f_func = sp.lambdify(self.x_sym, expr, 'numpy')
        d1_func = sp.lambdify(self.x_sym, d1_expr, 'numpy')
        d2_func = sp.lambdify(self.x_sym, d2_expr, 'numpy')

        try:
            y_val = float(f_func(value))
            d1_val = float(d1_func(value))
            d2_val = float(d2_func(value))
        except Exception as e:
            return None, f"Evaluation Error: {str(e)}"

        return {
            'y': y_val,
            'dy': d1_val,
            'd2y': d2_val,
            'expr': expr,
            'd1_expr': d1_expr,
            'd2_expr': d2_expr
        }, None

    def newton_forward_diff(self, value):
        df, expr, err = self.interp.newton_forward()
        if err: return None, None, err
        res, eval_err = self._evaluate_derivatives(expr, value)
        if eval_err: return None, None, eval_err
        return res, df, None

    def richardson_diff(self, value):
        """
        Richardson Extrapolation using Central Difference at step h and 2h.
        Formula: D = (4*D(h) - D(2h)) / 3
        """
        h = self.interp.check_equal_spacing()
        if h is None: return None, None, "Requires equal spacing."
        
        x_data = self.interp.x_data
        y_data = self.interp.y_data
        
        # Find nearest grid point
        idx = (np.abs(x_data - value)).argmin()
        
        if abs(x_data[idx] - value) > 1e-5:
            return None, None, "Richardson Extrapolation requires evaluating at a grid point."
            
        if idx < 2 or idx > len(x_data) - 3:
            return None, None, "Need at least 2 neighbors on each side (i-2 to i+2)."
            
        # 1st Derivative
        D_h = (y_data[idx+1] - y_data[idx-1]) / (2*h)
        D_2h = (y_data[idx+2] - y_data[idx-2]) / (4*h)
        dy = (4 * D_h - D_2h) / 3
        
        # 2nd Derivative (Central Diff error is also O(h^2), so same weights)
        D2_h = (y_data[idx+1] - 2*y_data[idx] + y_data[idx-1]) / (h**2)
        D2_2h = (y_data[idx+2] - 2*y_data[idx] + y_data[idx-2]) / ((2*h)**2)
        d2y = (4 * D2_h - D2_2h) / 3
        
        res = {
            "y": y_data[idx], "dy": dy, "d2y": d2y,
            "D_h": D_h, "D_2h": D_2h,
            "D2_h": D2_h, "D2_2h": D2_2h,
            "h": h
        }
        
        df_table = pd.DataFrame([
            {"Step": f"h={h:.4f}", "f'(x)": D_h, "f''(x)": D2_h},
            {"Step": f"2h={2*h:.4f}", "f'(x)": D_2h, "f''(x)": D2_2h},
            {"Step": "Richardson (Extrapolated)", "f'(x)": dy, "f''(x)": d2y}
        ])
        
        return res, df_table, None

    def cubic_spline_diff(self, value):
        """
        Uses the Cubic Spline coefficients to calculate derivatives.
        """
        df_coeffs, _, err = self.interp.cubic_spline_interpolation()
        if err: return None, None, err
        
        # Find the interval containing 'value'
        row = None
        start_x = 0
        
        for _, r in df_coeffs.iterrows():
            # Parse interval string "[1.0000, 2.0000]"
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", r['Interval'])
            if len(nums) >= 2:
                s, e = float(nums[0]), float(nums[1])
                # Allow slight tolerance for float comparison
                if s - 1e-9 <= value <= e + 1e-9:
                    row = r
                    start_x = s
                    break
        
        if row is None: return None, None, "Target x is outside the spline range."
        
        a, b, c, d = row['a'], row['b'], row['c'], row['d']
        dx = value - start_x
        
        # S(x) = a + b*dx + c*dx^2 + d*dx^3
        y_val = a + b*dx + c*dx**2 + d*dx**3
        dy = b + 2*c*dx + 3*d*dx**2
        d2y = 2*c + 6*d*dx
        
        res = {"y": y_val, "dy": dy, "d2y": d2y, "coeffs": row.to_dict(), "dx": dx}
        return res, df_coeffs, None

    def least_squares_diff(self, value):
        """
        Fits a parabola to the 5 nearest points and differentiates it.
        """
        x = self.interp.x_data
        y = self.interp.y_data
        
        # Find 5 nearest points
        dists = np.abs(x - value)
        idx_sorted = np.argsort(dists)
        k = min(len(x), 5)
        if k < 3: return None, None, "Need at least 3 points for Least Squares."
        
        indices = np.sort(idx_sorted[:k])
        x_sub, y_sub = x[indices], y[indices]
        
        # Fit y = Ax^2 + Bx + C
        A, B, C = np.polyfit(x_sub, y_sub, 2)
        
        y_val = A*value**2 + B*value + C
        dy = 2*A*value + B
        d2y = 2*A
        
        res = {"y": y_val, "dy": dy, "d2y": d2y, "coeffs": [A, B, C], "window_x": x_sub, "window_y": y_sub}
        
        # Create a table of the points used vs fitted
        y_fit = A*x_sub**2 + B*x_sub + C
        df_table = pd.DataFrame({
            "x": x_sub, "y_data": y_sub, "y_fit": y_fit, "Residual": y_sub - y_fit
        })
        
        return res, df_table, None

    def newton_backward_diff(self, value):
        df, expr, err = self.interp.newton_backward()
        if err: return None, None, err
        res, eval_err = self._evaluate_derivatives(expr, value)
        if eval_err: return None, None, eval_err
        return res, df, None

    def stirling_diff(self, value):
        df, expr, err = self.interp.stirling_method()
        if err: return None, None, err
        res, eval_err = self._evaluate_derivatives(expr, value)
        if eval_err: return None, None, eval_err
        return res, df, None

    def bessel_diff(self, value):
        df, expr, err = self.interp.bessel_method()
        if err: return None, None, err
        res, eval_err = self._evaluate_derivatives(expr, value)
        if eval_err: return None, None, eval_err
        return res, df, None

    def central_diff(self, value):
        # Uses Gauss Forward for standard Central Difference
        df, expr, err = self.interp.gauss_forward_method()
        if err: return None, None, err
        res, eval_err = self._evaluate_derivatives(expr, value)
        if eval_err: return None, None, eval_err
        return res, df, None

    def lagrange_derivative(self, value):
        df, expr = self.interp.lagrange_method()
        if expr is None:
            return None, df, "Failed to generate Lagrange polynomial."
        res, eval_err = self._evaluate_derivatives(expr, value)
        if eval_err: return None, df, eval_err
        return res, df, None

    def newton_divided_difference_diff(self, value):
        df, expr = self.interp.newton_divided_difference()
        if expr is None:
            return None, df, "Failed to generate Newton's Divided Difference polynomial."
        res, eval_err = self._evaluate_derivatives(expr, value)
        if eval_err: return None, df, eval_err
        return res, df, None

    def find_extrema(self):
        """
        Finds local minima and maxima within the data range by solving f'(x)=0.
        It uses a global Lagrange polynomial to represent the function.
        Returns: list of dicts [{'x': val, 'y': val, 'type': 'Maxima'/'Minima'}]
        """
        extrema = []
        try:
            # 1. Retrieve the global interpolating polynomial using Lagrange.
            # self.interp is already an InterpolationSolver instance.
            _, expr = self.interp.lagrange_method()
            
            if expr is None:
                return []

            # 2. Calculate the 1st and 2nd derivatives symbolically.
            f_prime = sp.diff(expr, self.x_sym, 1)
            f_double_prime = sp.diff(expr, self.x_sym, 2)
            
            # 3. Solve f'(x) = 0 by finding the roots of the derivative polynomial.
            poly_prime = sp.Poly(f_prime, self.x_sym)
            coeffs = poly_prime.all_coeffs()
            numeric_coeffs = [float(c) for c in coeffs]
            roots = np.roots(numeric_coeffs)
            
            # 4. Filter the roots and classify them as Minima or Maxima.
            x_min, x_max = min(self.interp.x_data), max(self.interp.x_data)
            
            for r in roots:
                # We only care about real roots (or complex with negligible imaginary part).
                if abs(r.imag) < 1e-9:
                    r_val = float(r.real)
                    
                    # Keep only roots that are inside the original data's x-range.
                    if x_min <= r_val <= x_max:
                        concavity = float(f_double_prime.subs(self.x_sym, r_val))
                        y_val = float(expr.subs(self.x_sym, r_val))
                        
                        # Classify using the second derivative test.
                        if concavity > 1e-6: # Concave up -> Local Minima
                            extrema.append({'x': r_val, 'y': y_val, 'type': 'Minima'})
                        elif concavity < -1e-6: # Concave down -> Local Maxima
                            extrema.append({'x': r_val, 'y': y_val, 'type': 'Maxima'})
                            
        except Exception as e:
            # This will print to the console if an error occurs during optimization.
            print(f"Error during optimization (find_extrema): {e}")
            
        return extrema