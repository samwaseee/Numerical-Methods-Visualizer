import numpy as np
import pandas as pd
import math

class ODESolver:
    def __init__(self, f_str, x0, y0, h, x_end):
        self.f_str = f_str
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.h = float(h)
        self.x_end = float(x_end)
        self.valid = False
        self.f = None
        self.error = None
        
        # Safe parsing context
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({
            "np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos, 
            "tan": np.tan, "sqrt": np.sqrt, "log": np.log, "pi": np.pi, "e": np.e
        })
        
        try:
            # Create lambda function
            # We wrap eval to catch errors during execution
            def func(x, y):
                return eval(f_str.replace("^", "**"), {"__builtins__": {}}, {**allowed_names, 'x': x, 'y': y})
            
            self.f = func
            # Test evaluation at initial point to validate
            self.f(x0, y0)
            self.valid = True
        except Exception as e:
            self.error = f"Invalid Equation: {str(e)}"

    def _get_x_vals(self):
        return np.arange(self.x0, self.x_end + self.h/1000.0, self.h)

    def evaluate_exact(self, exact_str, x_vals):
        """Evaluates an exact solution string y(x) for error calculation."""
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({
            "np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos, 
            "tan": np.tan, "sqrt": np.sqrt, "log": np.log, "pi": np.pi, "e": np.e
        })
        try:
            return eval(exact_str.replace("^", "**"), {"__builtins__": {}}, {**allowed_names, 'x': x_vals})
        except:
            return None

    def solve(self):
        if not self.valid:
            return None
        
        # Generate x values
        # Use a small epsilon to ensure x_end is included if it's a multiple of h
        x_vals = np.arange(self.x0, self.x_end + self.h/1000.0, self.h)
        n = len(x_vals)
        
        # Initialize arrays
        y_euler = np.zeros(n)
        y_heun = np.zeros(n)
        y_rk4 = np.zeros(n)
        
        # Set initial conditions
        y_euler[0] = self.y0
        y_heun[0] = self.y0
        y_rk4[0] = self.y0
        
        f = self.f
        h = self.h
        
        for i in range(n - 1):
            xi = x_vals[i]
            
            # --- Euler's Method ---
            yi_e = y_euler[i]
            try:
                slope_e = f(xi, yi_e)
                y_euler[i+1] = yi_e + h * slope_e
            except:
                y_euler[i+1] = np.nan
            
            # --- Heun's Method (Modified Euler) ---
            yi_h = y_heun[i]
            try:
                k1 = f(xi, yi_h)
                yp = yi_h + h * k1 # Predictor
                k2 = f(x_vals[i+1], yp)
                y_heun[i+1] = yi_h + (h / 2.0) * (k1 + k2) # Corrector
            except:
                y_heun[i+1] = np.nan

            # --- Runge-Kutta 4 (RK4) ---
            yi_r = y_rk4[i]
            try:
                k1 = f(xi, yi_r)
                k2 = f(xi + h/2.0, yi_r + h*k1/2.0)
                k3 = f(xi + h/2.0, yi_r + h*k2/2.0)
                k4 = f(xi + h, yi_r + h*k3)
                y_rk4[i+1] = yi_r + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            except:
                y_rk4[i+1] = np.nan
                
        df = pd.DataFrame({
            'x': x_vals,
            'y_euler': y_euler,
            'y_heun': y_heun,
            'y_rk4': y_rk4
        })
        
        return df

    def solve_euler(self):
        if not self.valid: return None, None
        x_vals = self._get_x_vals()
        n = len(x_vals)
        y = np.zeros(n)
        y[0] = self.y0
        f, h = self.f, self.h
        for i in range(n - 1):
            try: y[i+1] = y[i] + h * f(x_vals[i], y[i])
            except: y[i+1] = np.nan
        return x_vals, y

    def solve_heun(self):
        if not self.valid: return None, None
        x_vals = self._get_x_vals()
        n = len(x_vals)
        y = np.zeros(n)
        y[0] = self.y0
        f, h = self.f, self.h
        for i in range(n - 1):
            try:
                k1 = f(x_vals[i], y[i])
                yp = y[i] + h * k1
                k2 = f(x_vals[i+1], yp)
                y[i+1] = y[i] + (h / 2.0) * (k1 + k2)
            except: y[i+1] = np.nan
        return x_vals, y

    def solve_rk4(self):
        if not self.valid: return None, None
        x_vals = self._get_x_vals()
        n = len(x_vals)
        y = np.zeros(n)
        y[0] = self.y0
        f, h = self.f, self.h
        for i in range(n - 1):
            try:
                xi, yi = x_vals[i], y[i]
                k1, k2, k3, k4 = f(xi, yi), f(xi + 0.5*h, yi + 0.5*h*k1), f(xi + 0.5*h, yi + 0.5*h*k2), f(xi + h, yi + h*k3)
                y[i+1] = yi + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            except: y[i+1] = np.nan
        return x_vals, y

    def solve_rk45(self, tol=1e-6):
        """
        Adaptive Runge-Kutta-Fehlberg (RK45) Method.
        Returns a DataFrame with variable step sizes.
        """
        if not self.valid:
            return None
        
        f = self.f
        x = self.x0
        y = self.y0
        h = self.h if self.h > 0 else 0.1 # Initial step guess
        x_end = self.x_end
        
        x_vals = [x]
        y_vals = [y]
        
        # RKF45 Coefficients (Cash-Karp / Fehlberg)
        c2, c3, c4, c5, c6 = 1/4, 3/8, 12/13, 1, 1/2
        a21 = 1/4
        a31, a32 = 3/32, 9/32
        a41, a42, a43 = 1932/2197, -7200/2197, 7296/2197
        a51, a52, a53, a54 = 439/216, -8, 3680/513, -845/4104
        a61, a62, a63, a64, a65 = -8/27, 2, -3544/2565, 1859/4104, -11/40
        
        # 4th Order Solution
        b1, b3, b4, b5 = 25/216, 1408/2565, 2197/4104, -1/5
        # 5th Order Solution
        b5_1, b5_3, b5_4, b5_5, b5_6 = 16/135, 6656/12825, 28561/56430, -9/50, 2/55
        
        max_iter = 5000
        iter_count = 0
        
        while x < x_end and iter_count < max_iter:
            if x + h > x_end: h = x_end - x # Cap step at end

            try:
                k1 = f(x, y)
                k2 = f(x + c2*h, y + h*a21*k1)
                k3 = f(x + c3*h, y + h*(a31*k1 + a32*k2))
                k4 = f(x + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
                k5 = f(x + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
                k6 = f(x + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
                
                y_4 = y + h * (b1*k1 + b3*k3 + b4*k4 + b5*k5)
                y_5 = y + h * (b5_1*k1 + b5_3*k3 + b5_4*k4 + b5_5*k5 + b5_6*k6)
                
                error = abs(y_5 - y_4)
            except: break
            
            scale = 0.84 * (tol / (error + 1e-15))**0.25 if error > 0 else 2.0
            scale = min(max(scale, 0.1), 4.0) # Safety limits
            
            if error < tol: # Accept
                x += h; y = y_5
                x_vals.append(x); y_vals.append(y)
            
            h *= scale
            if h < 1e-10: h = 1e-10 # Prevent stall
            iter_count += 1
            
        return pd.DataFrame({'x': x_vals, 'y_rk45': y_vals})

class SystemODESolver:
    def __init__(self, func_strs, indep_var, dep_vars, t0, y0s, h, t_end):
        self.func_strs = func_strs
        self.indep_var = indep_var
        self.dep_vars = dep_vars
        self.t0 = float(t0)
        self.y0s = np.array(y0s, dtype=float)
        self.h = float(h)
        self.t_end = float(t_end)
        self.valid = False
        self.funcs = []
        self.error = None
        
        # Safe parsing context
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({
            "np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos, 
            "tan": np.tan, "sqrt": np.sqrt, "log": np.log, "pi": np.pi, "e": np.e
        })
        
        try:
            # Create lambda functions
            for f_str in func_strs:
                def make_func(s):
                    return lambda t, y_vec: eval(
                        s.replace("^", "**"), 
                        {"__builtins__": {}}, 
                        {**allowed_names, self.indep_var: t, **{name: val for name, val in zip(self.dep_vars, y_vec)}}
                    )
                self.funcs.append(make_func(f_str))
            
            # Test evaluation
            for f in self.funcs:
                f(self.t0, self.y0s)
            self.valid = True
        except Exception as e:
            self.error = f"Invalid Equation: {str(e)}"

    def solve(self):
        if not self.valid: return None
        
        t_vals = np.arange(self.t0, self.t_end + self.h/1000.0, self.h)
        n_steps = len(t_vals)
        n_vars = len(self.dep_vars)
        
        y_vals = np.zeros((n_steps, n_vars))
        y_vals[0] = self.y0s
        
        curr_y = self.y0s.copy()
        h = self.h
        
        for i in range(n_steps - 1):
            t = t_vals[i]
            
            # RK4 for systems
            k1 = np.array([f(t, curr_y) for f in self.funcs])
            
            y_k2 = curr_y + 0.5 * h * k1
            k2 = np.array([f(t + 0.5*h, y_k2) for f in self.funcs])
            
            y_k3 = curr_y + 0.5 * h * k2
            k3 = np.array([f(t + 0.5*h, y_k3) for f in self.funcs])
            
            y_k4 = curr_y + h * k3
            k4 = np.array([f(t + h, y_k4) for f in self.funcs])
            
            curr_y = curr_y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            y_vals[i+1] = curr_y
            
        data = {self.indep_var: t_vals}
        for j, var in enumerate(self.dep_vars):
            data[var] = y_vals[:, j]
            
        return pd.DataFrame(data)
