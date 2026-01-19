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
            
            # --- ADDED FOR HALLEY'S METHOD ---
            self.second_derivative = sp.diff(self.derivative, x)
            self.d2f = sp.lambdify(x, self.second_derivative, 'numpy')
            # ---------------------------------
            
            self.valid = True
        except:
            self.valid = False

    def get_latex(self):
        return sp.latex(self.expression)

    # --- NORMALIZATION LAYER ---
    def normalize_history(self, history, method_name):
        """Ensure all method histories have consistent core columns"""
        if not history:
            return history
        
        # Core columns every method should have
        required_cols = {'Iter', 'x_new', 'f(x)', 'Error % (Ea)'}
        
        normalized = []
        for row in history:
            normalized_row = dict(row)  # Copy row
            
            # Ensure core columns exist
            if 'Iter' not in normalized_row and len(normalized) >= 0:
                normalized_row['Iter'] = len(normalized) + 1
            
            if 'x_new' not in normalized_row:
                # Try to extract from method-specific keys
                normalized_row['x_new'] = normalized_row.get(
                    'x_new', 
                    normalized_row.get('c (root)', 
                    normalized_row.get('p_real', None))
                )
            
            if 'f(x)' not in normalized_row and 'x_new' in normalized_row:
                # Compute f(x) if missing
                try:
                    x_val = normalized_row['x_new']
                    if x_val is not None:
                        normalized_row['f(x)'] = self.f(x_val)
                except:
                    normalized_row['f(x)'] = None
            
            if 'Error % (Ea)' not in normalized_row:
                normalized_row['Error % (Ea)'] = 0.0
            
            normalized.append(normalized_row)
        
        return normalized

    # --- BENCHMARKING ENGINE ---
    def run_benchmark(self, x0, a, b, g_str, tol=0.01):
        results = []
        max_iter = 100

        def get_metrics(history):
            if not history: return max_iter, None, 100.0, "Failed"
            
            # Fallback to last
            last = history[-1]
            last_root = last.get('x_new', last.get('c (root)', 0))
            
            # Find EARLIEST valid convergence in the full list
            for step in history:
                try:
                    err = float(step.get('Error % (Ea)', 100))
                    root_val = step.get('x_new', step.get('c (root)', 0))
                    
                    # Compute f(root) for truth check
                    f_val = self.f(root_val) if root_val is not None else 1000.0
                    
                    if err is not None and err < tol and abs(f_val) < 0.1:
                        return step['Iter'], root_val, err, "Converged"
                except: pass
            
            return max_iter, last_root, last.get('Error % (Ea)', 100), "Slow/Diverged"

        # 1. Newton-Raphson
        try:
            nr_hist = self.newton_raphson_method(x0, max_iter)
            nr_hist = self.normalize_history(nr_hist, "Newton-Raphson")
            iters, root, err, status = get_metrics(nr_hist)
            results.append({"Method": "Newton-Raphson", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Newton-Raphson", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # 2. Bisection
        try:
            bi_hist = self.bisection_method(a, b, max_iter)
            bi_hist = self.normalize_history(bi_hist, "Bisection Method")
            iters, root, err, status = get_metrics(bi_hist)
            results.append({"Method": "Bisection Method", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Bisection Method", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # 3. Secant
        try:
            sec_hist = self.secant_method(x0, x0+1.0, max_iter)
            sec_hist = self.normalize_history(sec_hist, "Secant Method")
            iters, root, err, status = get_metrics(sec_hist)
            results.append({"Method": "Secant Method", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Secant Method", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # 4. False Position
        try:
            fp_hist = self.false_position_method(a, b, max_iter)
            fp_hist = self.normalize_history(fp_hist, "False Position")
            iters, root, err, status = get_metrics(fp_hist)
            results.append({"Method": "False Position", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "False Position", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # 5. Fixed Point
        try:
            fx_hist, _ = self.fixed_point_method(g_str, x0, max_iter)
            fx_hist = self.normalize_history(fx_hist, "Fixed Point")
            iters, root, err, status = get_metrics(fx_hist)
            results.append({"Method": "Fixed Point", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Fixed Point", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # --- NEW METHODS ---
        # 6. Brent's
        try:
            brent_hist = self.brents_method(a, b, max_iter)
            brent_hist = self.normalize_history(brent_hist, "Brent's Method")
            iters, root, err, status = get_metrics(brent_hist)
            results.append({"Method": "Brent's Method", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Brent's Method", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # 7. Halley's
        try:
            hal_hist = self.halleys_method(x0, max_iter)
            hal_hist = self.normalize_history(hal_hist, "Halley's Method")
            iters, root, err, status = get_metrics(hal_hist)
            results.append({"Method": "Halley's Method", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Halley's Method", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        # 8. Muller's
        try:
            mul_hist = self.mullers_method(x0, x0+0.5, x0+1.0, max_iter)
            mul_hist = self.normalize_history(mul_hist, "Muller's Method")
            iters, root, err, status = get_metrics(mul_hist)
            results.append({"Method": "Muller's Method", "Iterations": iters, "Root Found": root, "Error %": err, "Status": status})
        except: results.append({"Method": "Muller's Method", "Iterations": max_iter, "Status": "Failed", "Error %": 100})

        return pd.DataFrame(results).sort_values(by="Iterations")

    # --- NUMERICAL METHODS (PERSISTENT / NO-BREAK) ---

    def bisection_method(self, a, b, iterations):
        history = []
        curr_a, curr_b = a, b
        prev_c = None
        
        for i in range(iterations):
            c = (curr_a + curr_b) / 2
            try: f_c = self.f(c)
            except: f_c = 0
            
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
            
            if abs(f_x1 - f_x0) < 1e-12:
                new_x = curr_x1 
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
                
                if abs(f1) < 1e-9:
                    if not roots or abs(roots[-1] - x_curr) > 0.001:
                        roots.append(x_curr)
                        status = "Exact Root Found"
                        sign_change = "N/A (Hit Zero)"

                elif f1 * f2 < 0:
                    sign_change = "YES (Opposite Signs)"
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

    def brents_method(self, a, b, iterations, tol=1e-6):
        history = []
        try: fa, fb = self.f(a), self.f(b)
        except: return []
        
        # --- FIX: Check exact roots first ---
        if abs(fa) < 1e-12:
             row = {'Iter': 1, 'x_new': a, 'x_old': a, 'c (root)': a, 'f(x)': 0.0, 'Error % (Ea)': 0.0, 'Method Used': "Exact Root (a)"}
             for i in range(iterations): history.append({**row, 'Iter': i+1})
             return history
        if abs(fb) < 1e-12:
             row = {'Iter': 1, 'x_new': b, 'x_old': b, 'c (root)': b, 'f(x)': 0.0, 'Error % (Ea)': 0.0, 'Method Used': "Exact Root (b)"}
             for i in range(iterations): history.append({**row, 'Iter': i+1})
             return history

        # --- FIX: Invalid Bracket (Fill table with Error) ---
        if fa * fb > 0: 
            error_row = {'x_old': a, 'x_new': a, 'c (root)': a, 'f(x)': fa, 'Error % (Ea)': 100.0, 'Method Used': "No Bracket"}
            for i in range(iterations):
                history.append({**error_row, 'Iter': i+1})
            return history

        if abs(fa) < abs(fb): a, b = b, a; fa, fb = fb, fa
        c, fc = a, fa
        d = e = b - a
        mflag = True
        best_b = b

        for i in range(iterations):
            # Check convergence (freeze)
            if abs(fb) < 1e-12 or abs(b - a) < 1e-12:
                history.append({
                    'Iter': i+1, 'x_new': best_b, 'x_old': c, 'c (root)': best_b, 
                    'f(x)': self.f(best_b), 'Error % (Ea)': 0.0, 'Method Used': "Converged"
                })
                continue 

            if fa != fc and fb != fc:
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + (b * fa * fc) / ((fb - fa) * (fb - fc)) + (c * fa * fb) / ((fc - fa) * (fc - fb))
            else:
                s = b - fb * (b - a) / (fb - fa)
            
            cond1 = (s < (3 * a + b) / 4) or (s > b)
            cond2 = mflag and (abs(s - b) >= (abs(b - c) / 2))
            cond3 = not mflag and (abs(s - b) >= (abs(c - d) / 2))
            cond4 = mflag and (abs(b - c) < tol)
            cond5 = not mflag and (abs(c - d) < tol)

            if cond1 or cond2 or cond3 or cond4 or cond5:
                s = (a + b) / 2; mflag = True; method_used = "Bisection"
            else:
                mflag = False; method_used = "Interpolation"

            fs = self.f(s); d, c = e, b
            if fa * fs < 0: b, fb = s, fs
            else: a, fa = s, fs

            if abs(fa) < abs(fb): a, b = b, a; fa, fb = fb, fa
            best_b = b
            err = abs(b - a)
            
            history.append({
                'Iter': i+1, 'x_new': b, 'x_old': c, 'c (root)': b,
                'f(x)': fb, 'Error % (Ea)': err, 'Method Used': method_used
            })
        return history

    def halleys_method(self, x0, iterations):
        history = []
        curr_x = x0
        
        for i in range(iterations):
            fx = self.f(curr_x)
            dfx = self.df(curr_x)
            d2fx = self.d2f(curr_x)
            
            denom = 2 * dfx**2 - fx * d2fx
            
            # FREEZE logic
            if abs(denom) < 1e-12 or abs(fx) < 1e-12:
                new_x = curr_x
                ea = 0.0
            else:
                new_x = curr_x - (2 * fx * dfx) / denom
                ea = abs((new_x - curr_x) / new_x) * 100 if new_x != 0 else 0.0
            
            history.append({
                'Iter': i+1, 
                'x_old': curr_x, 
                'x_new': new_x, 
                'f(x)': fx, 
                "f'(x)": dfx,
                "f''(x)": d2fx,
                'Error % (Ea)': ea
            })
            
            curr_x = new_x
        return history

    def mullers_method(self, x0, x1, x2, iterations):
        history = []
        p0, p1, p2 = x0, x1, x2
        
        # Flag to track if convergence has already occurred
        converged = False
        final_state = {}

        for i in range(iterations):
            try:
                # If already converged, just append the final state repeatedly
                if converged:
                    # Update iteration count for the new row
                    current_row = final_state.copy()
                    current_row['Iter'] = i + 1
                    history.append(current_row)
                    continue

                f0, f1, f2 = self.f(p0), self.f(p1), self.f(p2)
                
                h1 = p1 - p0
                h2 = p2 - p1
                
                # Check for convergence based on function value
                if abs(f2) < 1e-10:
                    history.append({
                        'Iter': i+1, 
                        'p0': p0, 
                        'x_old': p2, 
                        'x_new': p2,  # Keep as complex if it is
                        'f(x)': f2,
                        'Error % (Ea)': 0.0,
                        'b_coef': 0, 'D': 0
                    })
                    converged = True
                    final_state = history[-1]
                    continue

                if abs(h1) < 1e-14 or abs(h2) < 1e-14: 
                    converged = True
                    final_state = history[-1] if history else {} # Fallback if immediate fail
                    continue

                d1 = (f1 - f0) / h1
                d2 = (f2 - f1) / h2
                d = (d2 - d1) / (h2 + h1)
                
                b_coef = d2 + h2 * d
                D_term = b_coef**2 - 4 * f2 * d
                
                # Handle complex square roots
                if isinstance(D_term, complex):
                    D = D_term**0.5
                else:
                    if D_term >= 0:
                        D = D_term**0.5
                    else:
                        D = (complex(D_term))**0.5
                
                # Choose denominator to maximize magnitude
                E1 = b_coef + D
                E2 = b_coef - D
                E = E1 if abs(E1) >= abs(E2) else E2
                
                if abs(E) < 1e-14:
                    h = 0
                else:
                    h = -2 * f2 / E
                
                p_new = p2 + h
                f_new = self.f(p_new)
                
                # Error calculation (works with complex too)
                if abs(p_new) > 1e-14:
                    ea = abs((p_new - p2) / p_new) * 100 
                else:
                    ea = abs(p_new - p2) * 100
                
                current_entry = {
                    'Iter': i+1, 
                    'p0': np.real(p0), 
                    'x_old': np.real(p2), 
                    'x_new': p_new,  # Keep as complex if needed
                    'f(x)': f_new,
                    'Error % (Ea)': ea,
                    'b_coef': b_coef, 'D': D
                }
                history.append(current_entry)
                
                # Check convergence
                if abs(f_new) < 1e-10 or ea < 1e-6:
                    converged = True
                    final_state = current_entry
                    continue
                
                p0, p1, p2 = p1, p2, p_new

            except Exception as e:
                # In case of error, if we have history, freeze on last known good state
                if history:
                    converged = True
                    final_state = history[-1]
                else:
                    break # catastrophic failure early on
            
        return history