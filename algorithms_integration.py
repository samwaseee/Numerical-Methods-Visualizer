import numpy as np

class IntegrationSolver:
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data, dtype=float)
        self.y_data = np.array(y_data, dtype=float)
        self.n_points = len(x_data)
        self.n_intervals = self.n_points - 1
        self.h = self._check_equal_spacing()

    def _check_equal_spacing(self):
        if self.n_points < 2: return None
        diffs = np.diff(self.x_data)
        h = diffs[0]
        if np.allclose(diffs, h, atol=1e-5):
            return h
        return None

    def trapezoidal_rule(self):
        if self.h is None: return None, "Requires equal spacing."
        
        y = self.y_data
        h = self.h
        
        # Formula: h/2 * [ (y0 + yn) + 2 * sum(others) ]
        sum_ends = y[0] + y[-1]
        sum_mid = np.sum(y[1:-1])
        result = (h / 2) * (sum_ends + 2 * sum_mid)
        
        details = {
            "h": h,
            "y0": y[0], "yn": y[-1],
            "sum_mid": sum_mid,
            "n": self.n_intervals
        }
        return result, details

    def simpson_13_rule(self):
        if self.h is None: return None, "Requires equal spacing."
        if self.n_intervals % 2 != 0: return None, "Simpson's 1/3 Rule requires an even number of intervals (odd number of points)."
        
        y = self.y_data
        h = self.h
        
        # Formula: h/3 * [ (y0 + yn) + 4*odd + 2*even ]
        sum_ends = y[0] + y[-1]
        sum_odd = np.sum(y[1:-1:2])
        sum_even = np.sum(y[2:-1:2])
        
        result = (h / 3) * (sum_ends + 4 * sum_odd + 2 * sum_even)
        
        details = {
            "h": h,
            "y0": y[0], "yn": y[-1],
            "sum_odd": sum_odd,
            "sum_even": sum_even,
            "n": self.n_intervals
        }
        return result, details

    def simpson_38_rule(self):
        if self.h is None: return None, "Requires equal spacing."
        if self.n_intervals % 3 != 0: return None, "Simpson's 3/8 Rule requires number of intervals to be a multiple of 3."
        
        y = self.y_data
        h = self.h
        n = self.n_intervals
        
        # Formula: 3h/8 * [ (y0 + yn) + 3*(others not multiple of 3) + 2*(multiples of 3) ]
        sum_ends = y[0] + y[-1]
        
        # Indices for multiples of 3 (excluding 0 and n)
        # 3, 6, 9 ... 
        idx_mult3 = [i for i in range(3, n, 3)]
        sum_mult3 = np.sum(y[idx_mult3]) if idx_mult3 else 0
        
        # Indices for others (excluding 0 and n)
        idx_others = [i for i in range(1, n) if i % 3 != 0]
        sum_others = np.sum(y[idx_others]) if idx_others else 0
        
        result = (3 * h / 8) * (sum_ends + 3 * sum_others + 2 * sum_mult3)
        
        details = {
            "h": h,
            "y0": y[0], "yn": y[-1],
            "sum_others": sum_others,
            "sum_mult3": sum_mult3,
            "n": self.n_intervals
        }
        return result, details

    def booles_rule(self):
        if self.h is None: return None, "Requires equal spacing."
        if self.n_intervals % 4 != 0: return None, "Boole's Rule requires number of intervals to be a multiple of 4."
        
        y = self.y_data
        h = self.h
        n = self.n_intervals
        
        sum_ends = y[0] + y[-1]
        sum_32, sum_12, sum_14 = 0, 0, 0
        
        for i in range(1, n):
            if i % 4 == 0:
                sum_14 += y[i]
            elif i % 2 == 0:
                sum_12 += y[i]
            else: # odd
                sum_32 += y[i]
                
        result = (2 * h / 45) * (7 * sum_ends + 32 * sum_32 + 12 * sum_12 + 14 * sum_14)
        
        details = {
            "h": h,
            "y0": y[0], "yn": y[-1],
            "sum_32": sum_32,
            "sum_12": sum_12,
            "sum_14": sum_14,
            "n": self.n_intervals
        }
        return result, details

    def weddles_rule(self):
        if self.h is None: return None, "Requires equal spacing."
        if self.n_intervals % 6 != 0: return None, "Weddle's Rule requires number of intervals to be a multiple of 6."
        
        y = self.y_data
        h = self.h
        n = self.n_intervals
        
        sum_ends = y[0] + y[-1]
        sum_5, sum_1, sum_6, sum_2 = 0, 0, 0, 0
        
        for i in range(1, n):
            if i % 6 == 1 or i % 6 == 5:
                sum_5 += y[i]
            elif i % 6 == 2 or i % 6 == 4:
                sum_1 += y[i]
            elif i % 6 == 3:
                sum_6 += y[i]
            elif i % 6 == 0:
                sum_2 += y[i]
                
        result = (3 * h / 10) * (sum_ends + 5*sum_5 + sum_1 + 6*sum_6 + 2*sum_2)
        
        details = {
            "h": h,
            "y0": y[0], "yn": y[-1],
            "sum_5": sum_5,
            "sum_1": sum_1,
            "sum_6": sum_6,
            "sum_2": sum_2,
            "n": self.n_intervals
        }
        return result, details
