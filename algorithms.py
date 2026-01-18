import sympy as sp
import numpy as np
import pandas as pd

class EquationSolver:
    def __init__(self, func_str):
        self.original_str = func_str
        clean_str = func_str.replace("^", "**")
        
        x = sp.symbols('x')
        try:
            self.expression = sp.sympify(clean_str)
            self.f = sp.lambdify(x, self.expression, 'numpy')
            self.derivative = sp.diff(self.expression, x)
            self.df = sp.lambdify(x, self.derivative, 'numpy')
            self.valid = True
        except:
            self.valid = False

    def get_latex(self):
        return sp.latex(self.expression)

    # --- BENCHMARKING ENGINE ---
    def run_benchmark(self, x0, a, b, g_str, tol=0.01):
        results = []
        max_iter = 100

        def get_metrics(history):
            if not history: return max_iter, None, 100.0, "Failed"
            
            # Fallback to last
            last = history[-1]
            last_root = last.get('x_new', last.get('c (root)'))
            
            # Find EARLIEST valid convergence in the full list
            for step in history:
                try:
                    err = float(step.get('Error % (Ea)', 100))
                    root_val = step.get('x_new', step.get('c (root)'))
                    
                    # Compute f(root) for truth check
                    f_val = self.f(root_val) if root_val is not None else 1000.0
                    
                    if err is not None and err < tol and abs(f_val) < 0.1:
                        return step['Iter'], root_val, err, "Converged"
                except: pass
            
            return max_iter, last_root, last.get('Error % (Ea)', 100), "Slow/Diverged"

        # 1. Newton-Raphson
        try:
            nr_hist = self.newton_raphson_method(x0, max_iter)
            iters, root, err, status = get_metrics(nr_hist)
            results.append({"Method": "Newton-Raphson", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Newton-Raphson", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # 2. Bisection
        try:
            bi_hist = self.bisection_method(a, b, max_iter)
            iters, root, err, status = get_metrics(bi_hist)
            results.append({"Method": "Bisection Method", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Bisection Method", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # 3. Secant
        try:
            sec_hist = self.secant_method(x0, x0+1.0, max_iter)
            iters, root, err, status = get_metrics(sec_hist)
            results.append({"Method": "Secant Method", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Secant Method", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # 4. False Position
        try:
            fp_hist = self.false_position_method(a, b, max_iter)
            iters, root, err, status = get_metrics(fp_hist)
            results.append({"Method": "False Position", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "False Position", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # 5. Fixed Point
        try:
            fx_hist, _ = self.fixed_point_method(g_str, x0, max_iter)
            iters, root, err, status = get_metrics(fx_hist)
            results.append({"Method": "Fixed Point", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Fixed Point", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        return pd.DataFrame(results).sort_values(by="Iterations")

    # --- NUMERICAL METHODS (PERSISTENT / NO-BREAK) ---

    def bisection_method(self, a, b, iterations):
        history = []
        curr_a, curr_b = a, b
        prev_c = None
        
        # NOTE: Bisection naturally stops updating if bracket is tiny, 
        # but we let the loop run to fill the table.
        for i in range(iterations):
            c = (curr_a + curr_b) / 2
            f_c = self.f(c)
            ea = abs((c - prev_c) / c) * 100 if prev_c and c != 0 else None
            
            history.append({"Iter": i+1, "a": curr_a, "b": curr_b, "c (root)": c, "f(c)": f_c, "Error % (Ea)": ea})
            
            if abs(f_c) < 1e-12:
                pass 
            else:
                if self.f(curr_a) * f_c < 0:
                    curr_b = c
                else:
                    curr_a = c
            prev_c = c
        return history

    def false_position_method(self, a, b, iterations):
        history = []
        curr_a, curr_b = a, b
        prev_c = None
        
        for i in range(iterations):
            fa, fb = self.f(curr_a), self.f(curr_b)
            
            # Avoid division by zero if bracket collapses
            if abs(fb - fa) < 1e-12:
                c = curr_a
            else:
                c = (curr_a * fb - curr_b * fa) / (fb - fa)
                
            f_c = self.f(c)
            ea = abs((c - prev_c) / c) * 100 if prev_c and c != 0 else None
            
            history.append({"Iter": i+1, "a": curr_a, "b": curr_b, "c (root)": c, "f(c)": f_c, "Error % (Ea)": ea})
            
            if abs(f_c) < 1e-12: pass
            elif self.f(curr_a) * f_c < 0:
                curr_b = c
            else:
                curr_a = c
            prev_c = c
        return history

    def fixed_point_method(self, g_str, x0, iterations):
        history = []
        x = sp.symbols('x')
        try:
            g_expr = sp.sympify(g_str.replace("^", "**"))
            g = sp.lambdify(x, g_expr, 'numpy')
        except: return [], None
        
        curr_x = x0
        for i in range(iterations):
            try:
                new_x = g(curr_x)
                if np.isnan(new_x) or np.isinf(new_x): 
                    new_x = curr_x # Freeze on error
            except: 
                new_x = curr_x
            
            ea = abs((new_x - curr_x) / new_x) * 100 if new_x != 0 else 0.0
            current_f = self.f(new_x)
            
            history.append({
                "Iter": i+1, 
                "x_old": curr_x, 
                "x_new": new_x, 
                "f(x)": current_f, 
                "Error % (Ea)": ea, 
                "g(x)": new_x
            })
            
            curr_x = new_x
            
        return history, g

    def newton_raphson_method(self, x0, iterations):
        history = []
        curr_x = x0
        for i in range(iterations):
            fx = self.f(curr_x)
            dfx = self.df(curr_x)
            
            # Persist if converged (dfx=0 or fx=0)
            if abs(dfx) < 1e-12:
                new_x = curr_x
                ea = 0.0
            else:
                try:
                    new_x = curr_x - (fx / dfx)
                    ea = abs((new_x - curr_x) / new_x) * 100 if new_x != 0 else 0.0
                except:
                    new_x = curr_x
                    ea = 0.0
            
            history.append({"Iter": i+1, "x_old": curr_x, "x_new": new_x, "f(x)": fx, "f'(x)": dfx, "Error % (Ea)": ea})
            curr_x = new_x
        return history

    def secant_method(self, x0, x1, iterations):
        history = []
        curr_x0, curr_x1 = x0, x1
        
        for i in range(iterations):
            f_x0 = self.f(curr_x0)
            f_x1 = self.f(curr_x1)
            
            # Persist if converged (denominator approx 0)
            if abs(f_x1 - f_x0) < 1e-12:
                new_x = curr_x1 # Stay at best root
                ea = 0.0
            else:
                try:
                    new_x = curr_x1 - (f_x1 * (curr_x1 - curr_x0)) / (f_x1 - f_x0)
                    ea = abs((new_x - curr_x1) / new_x) * 100 if new_x != 0 else 0.0
                except:
                    new_x = curr_x1
                    ea = 0.0

            history.append({
                "Iter": i+1, 
                "x_prev": curr_x0, 
                "x_curr": curr_x1, 
                "x_new": new_x,
                "f(x)": self.f(new_x),
                "Error % (Ea)": ea
            })
            
            curr_x0, curr_x1 = curr_x1, new_x
            
        return history

    # --- SCANNER ---
    def find_all_roots(self, start, end, step_size=0.1):
        roots = []
        scan_log = []
        x_curr = start
        
        while x_curr < end:
            x_next = x_curr + step_size
            status = "No Root"
            sign_change = "No"
            f1 = 0
            f2 = 0
            
            try:
                f1 = self.f(x_curr)
                f2 = self.f(x_next)
                
                # Check 1: Exact root
                if abs(f1) < 1e-9:
                    if not roots or abs(roots[-1] - x_curr) > 0.001:
                        roots.append(x_curr)
                        status = "Exact Root Found"
                        sign_change = "N/A (Hit Zero)"

                # Check 2: Sign Change (IVT)
                elif f1 * f2 < 0:
                    sign_change = "YES (Opposite Signs)"
                    # Run Bisection to pinpoint
                    r_hist = self.bisection_method(x_curr, x_next, 20)
                    if r_hist:
                        root = r_hist[-1]['c (root)']
                        if not roots or abs(roots[-1] - root) > 0.001:
                            roots.append(root)
                            status = f"Converged {root:.4f}"
            except: pass
            
            scan_log.append({
                "Interval Start": x_curr,
                "Interval End": x_next,
                "f(Start)": f1,
                "f(End)": f2,
                "Sign Change?": sign_change,
                "Status": status
            })
            
            x_curr = x_next
            
        return roots, scan_log