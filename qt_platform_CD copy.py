import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QTextEdit,
    QPushButton, QMessageBox, QHBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize


def generate_best_grid_points(bounds, cost_expr, constraint_exprs, S, step=0.1):
    # Generate grid points for each variable
    grids = [np.arange(low, high + step, step) for (low, high) in bounds]
    mesh = np.meshgrid(*grids, indexing='xy')
    candidates = np.vstack([m.flatten() for m in mesh]).T
    
    # Prepare constraint functions
    constraint_funcs = []
    for expr in constraint_exprs:
        if expr.strip():
            transformed = parse_constraint_expression(expr.strip())
            constraint_funcs.append(lambda x, e=transformed: eval(e, {}, {"x": x}) >= 0)

    valid_points = []
    for point in candidates:
        # Check constraints
        if all(f(point) for f in constraint_funcs):
            cost_val = eval(cost_expr, {}, {"x": point})
            valid_points.append((cost_val, point))
    
    # Sort by cost and take top S
    valid_points.sort(key=lambda t: t[0])
    return [p for _, p in valid_points[:S]]


def translate_vars(expr):
    """
    Convert variables x1, x2, ... to zero-based x[0], x[1], ...
    """
    return re.sub(r'\bx(\d+)\b', lambda m: f"x[{int(m.group(1))-1}]", expr)


def parse_constraint_expression(expr):
    expr = translate_vars(expr)
    if "<=" in expr:
        left, right = expr.split("<=")
        return f"({right}) - ({left})"
    elif ">=" in expr:
        left, right = expr.split(">=")
        return f"({left}) - ({right})"
    elif "<" in expr:
        left, right = expr.split("<")
        return f"({right}) - ({left}) - 1e-9"
    elif ">" in expr:
        left, right = expr.split(">")
        return f"({left}) - ({right}) - 1e-9"
    else:
        raise ValueError(f"Invalid constraint format (need <=, >=, <, or >): {expr}")


def solve_from_initial_point(x0, cost_expr, constraint_exprs, bounds):
    cost_values = []

    def cost_fn(x):
        val = eval(cost_expr, {}, {"x": x})
        cost_values.append(float(val))
        return float(val)

    cons = []
    for expr in constraint_exprs:
        s = expr.strip()
        if not s:
            continue
        transformed = parse_constraint_expression(s)

        def make_fun(trans_expr):
            return lambda x: float(eval(trans_expr, {}, {"x": x}))

        cons.append({"type": "ineq", "fun": make_fun(transformed)})

    result = minimize(cost_fn, x0, method="SLSQP", bounds=bounds, constraints=cons)
    result.cost_trace = cost_values
    return result


class OptimizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-inital Coordinate Descent Optimization (x1, x2, ...)")
        self.setGeometry(100, 100, 900, 720)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Number of variables (n):"))
        self.num_vars_input = QLineEdit("3")
        top_row.addWidget(self.num_vars_input)

        top_row.addWidget(QLabel("Number of initial points:"))
        self.num_init_input = QLineEdit("8")
        top_row.addWidget(self.num_init_input)

        self.bounds_input = QTextEdit()
        self.bounds_input.setPlaceholderText(
            "Examples:\n"
            "x1: -5, 5\n"
            "x2: 0, 10\n"
            "x3: 1        # shorthand -> [-1,1]\n"
            "all: -5, 5\n"
            "x1-x3: -2, 2\n" 
            "You may separate multiple entries with ';' on the same line."
        )

        self.cost_input = QLineEdit()
        self.cost_input.setPlaceholderText("e.g. (x1-1)**2 + (x2+2)**2 + 0.5*(x3)**2")

        self.constraints_input = QTextEdit()
        self.constraints_input.setPlaceholderText(
            "Constraints, comma-separated or newline separated. Use x1, x2, ...\n"
            "Example:\n x1 >= 0, x2 >= -2, x1 + x2 <= 5"
        )

        self.solve_button = QPushButton("Solve (parallel)")
        self.solve_button.clicked.connect(self.solve_optimization)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)

        self.plot_canvas = FigureCanvas(plt.figure())

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(QLabel("Bounds (variable: low, high):"))
        layout.addWidget(self.bounds_input)
        layout.addWidget(QLabel("Cost function (use x1, x2, ...):"))
        layout.addWidget(self.cost_input)
        layout.addWidget(QLabel("Constraints (use x1, x2, ...). Separate by comma or new line:"))
        layout.addWidget(self.constraints_input)
        layout.addWidget(self.solve_button)
        layout.addWidget(QLabel("Optimization Results:"))
        layout.addWidget(self.result_output)
        layout.addWidget(QLabel("Best cost trace (function evaluations):"))
        layout.addWidget(self.plot_canvas)

        self.setLayout(layout)

    def _parse_bounds(self, n):
        default_bound = (-1e6, 1e6)
        bounds = [default_bound] * n
        text = self.bounds_input.toPlainText().strip()
        if not text:
            return bounds

        raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        entries = []
        for ln in raw_lines:
            parts = [p.strip() for p in re.split(r';|\|', ln) if p.strip()]
            entries.extend(parts)

        for entry in entries:
            if ':' not in entry:
                raise ValueError(f"Bad bound format in line: {entry}")
            idx_str, rng = entry.split(":", 1)
            idx_str = idx_str.strip().lower()
            rng = rng.strip()

            if idx_str == 'all':
                idxs = list(range(n))
            elif '-' in idx_str:  # x1-x3
                a, b = idx_str.split('-', 1)
                ia = self._var_to_index(a.strip())
                ib = self._var_to_index(b.strip())
                if ia > ib:
                    raise ValueError(f"Invalid index range in line: {entry}")
                idxs = list(range(ia, ib + 1))
            else:
                idxs = [self._var_to_index(idx_str)]

            nums = [tok for tok in re.split(r'[,\s]+', rng) if tok != ""]
            if len(nums) == 2:
                low = float(nums[0])
                high = float(nums[1])
            elif len(nums) == 1:
                val = float(nums[0])
                low = -abs(val)
                high = abs(val)
            else:
                raise ValueError(f"Bad bound format in line: {entry}")

            for idx in idxs:
                if idx < 0 or idx >= n:
                    raise ValueError(f"Index {idx} out of range for n={n}")
                bounds[idx] = (low, high)

        return bounds

    def _var_to_index(self, var):
        if var.startswith('x'):
            return int(var[1:]) - 1
        else:
            return int(var)  # fallback

    def solve_optimization(self):
        try:
            n = int(self.num_vars_input.text().strip())
            if n <= 0:
                raise ValueError("Number of variables must be a positive integer.")
            num_init = int(self.num_init_input.text().strip())
            if num_init <= 0:
                raise ValueError("Number of initial points must be a positive integer.")

            bounds = self._parse_bounds(n)
            cost_expr = translate_vars(self.cost_input.text().strip())
            if not cost_expr:
                raise ValueError("Provide a cost function (uses x1, x2, ...).")

            raw_cons = self.constraints_input.toPlainText().strip()
            if raw_cons == "":
                constraint_list = []
            else:
                parts = [p.strip() for p in re.split(r'[,;\n]+', raw_cons) if p.strip()]
                constraint_list = parts

            # initial_points = []
            # for _ in range(num_init):
            #     xi = np.array([np.random.uniform(low, high) for (low, high) in bounds], dtype=float)
            #     initial_points.append(xi)
            initial_points = generate_best_grid_points(bounds, cost_expr, constraint_list, num_init, step=0.01)


            self.result_output.clear()
            self.result_output.append("Running parallel optimizations...\n")
            results = []

            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(solve_from_initial_point, x0, cost_expr, constraint_list, bounds)
                    for x0 in initial_points
                ]
                for idx, future in enumerate(as_completed(futures), start=1):
                    try:
                        res = future.result()
                        if res.success:
                            results.append(res)
                            self.result_output.append(
                                f"[Init {idx}] Success: cost={res.fun:.6g}, x={self._format_solution(res.x)}"
                            )
                        else:
                            self.result_output.append(f"[Init {idx}] Failed: {res.message}")
                    except Exception as e:
                        self.result_output.append(f"[Init {idx}] Error: {str(e)}")

            if not results:
                self.result_output.append("\nNo successful runs found.")
                return

            best = min(results, key=lambda r: r.fun)
            self.result_output.append("\n=== Best Solution ===")
            for i, val in enumerate(best.x):
                self.result_output.append(f"x{i+1} = {float(val):.8g}")
            self.result_output.append(f"Minimum cost: {best.fun:.12g}")
            self.result_output.append(f"Number of cost evaluations (best run): {len(best.cost_trace)}")

            self._plot_cost_trace(best.cost_trace)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _format_solution(self, arr):
        return "[" + ", ".join(f"x{i+1}={v:.6g}" for i, v in enumerate(arr)) + "]"

    def _plot_cost_trace(self, trace):
        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)
        ax.plot(trace, marker='o')
        ax.set_title("Cost evaluations (best run)")
        ax.set_xlabel("Function evaluation index")
        ax.set_ylabel("Cost")
        ax.grid(True)
        self.plot_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OptimizationApp()
    win.show()
    sys.exit(app.exec_())
