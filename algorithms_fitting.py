import numpy as np

class FittingSolver:
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data, dtype=float)
        self.y_data = np.array(y_data, dtype=float)
        self.n = len(x_data)

    def _calculate_r2(self, y_pred):
        """Calculates the Coefficient of Determination (R^2)"""
        ss_res = np.sum((self.y_data - y_pred) ** 2)
        ss_tot = np.sum((self.y_data - np.mean(self.y_data)) ** 2)
        if ss_tot == 0: return 0.0
        return 1 - (ss_res / ss_tot)

    def linear_regression(self):
        """Fits y = mx + c"""
        if self.n < 2: return {"error": "Need at least 2 points for linear regression."}
        
        # 1. Calculate Sums for Normal Equations
        n = self.n
        sx = np.sum(self.x_data)
        sy = np.sum(self.y_data)
        sxx = np.sum(self.x_data**2)
        sxy = np.sum(self.x_data * self.y_data)
        
        coeffs = np.polyfit(self.x_data, self.y_data, 1)
        m, c = coeffs
        
        func = np.poly1d(coeffs)
        y_pred = func(self.x_data)
        r2 = self._calculate_r2(y_pred)
        mse = np.mean((self.y_data - y_pred)**2)
        
        sign = "+" if c >= 0 else "-"
        equation = f"y = {m:.4f}x {sign} {abs(c):.4f}"
        
        return {
            "coeffs": coeffs,
            "equation": equation,
            "r2": r2,
            "mse": mse,
            "predict": func,
            "type": "Linear Regression",
            "steps": {
                "n": n, "sx": sx, "sy": sy, "sxx": sxx, "sxy": sxy,
                "a0": c, "a1": m
            }
        }

    def polynomial_regression(self, degree):
        """Fits polynomial of given degree"""
        if self.n <= degree:
            return {"error": f"Need at least {degree + 1} points for degree {degree} polynomial."}
            
        coeffs = np.polyfit(self.x_data, self.y_data, degree)
        func = np.poly1d(coeffs)
        y_pred = func(self.x_data)
        r2 = self._calculate_r2(y_pred)
        mse = np.mean((self.y_data - y_pred)**2)
        
        # Format Equation
        terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if abs(c) < 1e-5: continue # Skip tiny coefficients
            
            val_str = f"{abs(c):.4f}"
            sign = " + " if c >= 0 else " - "
            if i == 0: sign = "" if c >= 0 else "-"
            
            if power == 0:
                terms.append(f"{sign}{val_str}")
            elif power == 1:
                terms.append(f"{sign}{val_str}x")
            else:
                terms.append(f"{sign}{val_str}x^{power}")
                
        equation = "y = " + "".join(terms)
        
        # Prepare Normal Equation Matrix for display
        mat_size = degree + 1
        matrix = np.zeros((mat_size, mat_size))
        rhs = np.zeros(mat_size)
        
        x_powers = {p: np.sum(self.x_data**p) for p in range(2*degree + 1)}
        
        for r in range(mat_size):
            for c in range(mat_size):
                matrix[r, c] = x_powers[r + c]
            rhs[r] = np.sum((self.x_data**r) * self.y_data)
        
        return {
            "coeffs": coeffs,
            "equation": equation,
            "r2": r2,
            "mse": mse,
            "predict": func,
            "type": f"Polynomial (Degree {degree})",
            "steps": {"matrix": matrix.tolist(), "rhs": rhs.tolist(), "degree": degree}
        }

    def exponential_fit(self):
        """Fits y = a * e^(bx) via Linearization ln(y) = ln(a) + bx"""
        if np.any(self.y_data <= 0):
            return {"error": "Exponential fit requires all Y values to be positive (> 0)."}
            
        # Linearize: Y = A + Bx where Y=ln(y), A=ln(a), B=b
        y_log = np.log(self.y_data)
        
        # Calculate sums for the linearized system
        n = self.n
        sx = np.sum(self.x_data)
        sy_log = np.sum(y_log)
        sxx = np.sum(self.x_data**2)
        sxy_log = np.sum(self.x_data * y_log)
        
        coeffs = np.polyfit(self.x_data, y_log, 1)
        b, ln_a = coeffs
        a = np.exp(ln_a)
        
        def func(x):
            # Handle scalar or array input
            x_arr = np.array(x) if isinstance(x, (list, tuple)) else x
            return a * np.exp(b * x_arr)
            
        y_pred = func(self.x_data)
        r2 = self._calculate_r2(y_pred)
        mse = np.mean((self.y_data - y_pred)**2)
        
        equation = f"y = {a:.4f}e^{{{b:.4f}x}}"
        
        return {
            "coeffs": [a, b],
            "equation": equation,
            "r2": r2,
            "mse": mse,
            "predict": func,
            "type": "Exponential Fit",
            "steps": {
                "n": n, "sx": sx, "sy_log": sy_log, 
                "sxx": sxx, "sxy_log": sxy_log,
                "A": ln_a, "B": b,
                "a": a, "b": b
            }
        }
