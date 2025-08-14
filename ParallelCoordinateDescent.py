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


def parse_constraint_expression(expr):
    """
    Convert human comparisons into numeric-in-equality expressions suitable for scipy 'ineq' constraints.
    Input expressions should reference the optimization vector as x[0], x[1], ...
    Examples:
      "x[0] + x[1] <= 5" -> "(5) - (x[0] + x[1])"
      "x[2] >= -1"       -> "(x[2]) - (-1)"
    """
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
    """
    Worker function (runs inside each process).
    - cost_expr is a string that may use x[0], x[1], ...
    - constraint_exprs is a list of strings like 'x[0] + x[1] <= 5'
    Returns a scipy OptimizeResult with attribute 'cost_trace' appended.
    """
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
        # create function that evaluates transformed expression using 'x'
        def make_fun(trans_expr):
            return lambda x: float(eval(trans_expr, {}, {"x": x}))
        cons.append({"type": "ineq", "fun": make_fun(transformed)})

    result = minimize(cost_fn, x0, method="SLSQP", bounds=bounds, constraints=cons)
    result.cost_trace = cost_values
    return result


class OptimizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parallel Vector Optimization (x[0], x[1], ...)")
        self.setGeometry(100, 100, 900, 1000)

        # UI elements
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Number of variables (n):"))
        self.num_vars_input = QLineEdit("3")
        top_row.addWidget(self.num_vars_input)

        top_row.addWidget(QLabel("Number of initial points:"))
        self.num_init_input = QLineEdit("8")
        top_row.addWidget(self.num_init_input)

        self.bounds_input = QTextEdit()
        self.bounds_input.setPlaceholderText(
            "Index: low, high   (one per line)  Example:\n"
            "0: -5, 5\n"
            "1: 0, 10\n"
            "2: 1        # shorthand -> [-1,1]\n"
            "all: -5, 5\n"
            "0-3: -2, 2\n"
            "You may separate multiple entries with ';' on the same line."
        )

        self.cost_input = QLineEdit()
        self.cost_input.setPlaceholderText("e.g. (x[0]-1)**2 + (x[1]+2)**2 + 0.5*(x[2])**2")

        self.constraints_input = QTextEdit()
        self.constraints_input.setPlaceholderText(
            "Constraints, comma-separated or newline separated. Use x[i] notation.\n"
            "Example:\n x[0] >= 0, x[1] >= -2, x[0] + x[1] <= 5"
        )

        self.solve_button = QPushButton("Solve (parallel)")
        self.solve_button.clicked.connect(self.solve_optimization)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)

        self.plot_canvas = FigureCanvas(plt.figure())

        # Layout assembly
        layout = QVBoxLayout()
        layout.addLayout(top_row)

        layout.addWidget(QLabel("Bounds (index: low, high). Single-number shorthand interpreted as symmetric bound [-val, val]:"))
        layout.addWidget(self.bounds_input)

        layout.addWidget(QLabel("Cost function (use x[0], x[1], ...):"))
        layout.addWidget(self.cost_input)

        layout.addWidget(QLabel("Constraints (use x[0], x[1], ...). Separate by comma or new line:"))
        layout.addWidget(self.constraints_input)

        layout.addWidget(self.solve_button)
        layout.addWidget(QLabel("Optimization Results:"))
        layout.addWidget(self.result_output)
        layout.addWidget(QLabel("Best cost trace (function evaluations):"))
        layout.addWidget(self.plot_canvas)

        self.setLayout(layout)

    def _parse_bounds(self, n):
        """
        Parse bounds text into list of length n of (low, high).
        Accept lines like:
          0: -5, 5
          1: -2 2
          2: 1         # shorthand -> [-1, 1]
          all: -3, 3
          0-3: -2, 2
        Use ';' to separate multiple entries on same line.
        """
        default_bound = (-1e6, 1e6)
        bounds = [default_bound] * n
        text = self.bounds_input.toPlainText().strip()
        if not text:
            return bounds

        # Split into entries by newline, and allow ';' to separate multiple entries on same line
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
            # parse indices: all, single index, or range "a-b"
            if idx_str == 'all':
                idxs = list(range(n))
            elif '-' in idx_str:
                a, b = idx_str.split('-', 1)
                ia = int(a.strip()); ib = int(b.strip())
                if ia > ib:
                    raise ValueError(f"Invalid index range in line: {entry}")
                idxs = list(range(ia, ib + 1))
            else:
                idxs = [int(idx_str)]

            # parse range numbers: try splitting by comma first, then whitespace
            # allow single number shorthand
            nums = [tok for tok in re.split(r'[,\s]+', rng) if tok != ""]
            if len(nums) == 2:
                low = float(nums[0]); high = float(nums[1])
            elif len(nums) == 1:
                val = float(nums[0])
                # interpret single number as symmetric bound [-val, val]
                low = -abs(val); high = abs(val)
            else:
                raise ValueError(f"Bad bound format in line: {entry}")

            for idx in idxs:
                if idx < 0 or idx >= n:
                    raise ValueError(f"Index {idx} out of range for n={n}")
                bounds[idx] = (low, high)

        return bounds

    def solve_optimization(self):
        try:
            n = int(self.num_vars_input.text().strip())
            if n <= 0:
                raise ValueError("Number of variables must be a positive integer.")
            num_init = int(self.num_init_input.text().strip())
            if num_init <= 0:
                raise ValueError("Number of initial points must be a positive integer.")

            bounds = self._parse_bounds(n)
            cost_expr = self.cost_input.text().strip()
            if not cost_expr:
                raise ValueError("Provide a cost function (uses x[0], x[1], ...).")

            raw_cons = self.constraints_input.toPlainText().strip()
            if raw_cons == "":
                constraint_list = []
            else:
                parts = [p.strip() for p in re.split(r'[,;\n]+', raw_cons) if p.strip()]
                constraint_list = parts

            initial_points = []
            for _ in range(num_init):
                xi = np.array([np.random.uniform(low, high) for (low, high) in bounds], dtype=float)
                initial_points.append(xi)

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
                            self.result_output.append(f"[Init {idx}] Success: cost={res.fun:.6g}, x={np.array2string(res.x, precision=6)}")
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
                self.result_output.append(f"x[{i}] = {float(val):.8g}")
            self.result_output.append(f"Minimum cost: {best.fun:.12g}")
            self.result_output.append(f"Number of cost evaluations (best run): {len(best.cost_trace)}")

            self._plot_cost_trace(best.cost_trace)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

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
