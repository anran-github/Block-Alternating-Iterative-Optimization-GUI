import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QTextEdit,
    QPushButton, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize


def parse_constraint_expression(expr):
    """
    Converts logical expression into numerical inequality for scipy
    E.g., 'x + y <= 5' -> '5 - (x + y)'
          'x >= 0'     -> 'x - 0'
    """
    if "<=" in expr:
        left, right = expr.split("<=")
        return f"({right}) - ({left})"
    elif ">=" in expr:
        left, right = expr.split(">=")
        return f"({left}) - ({right})"
    elif "<" in expr:
        left, right = expr.split("<")
        return f"({right}) - ({left}) - 1e-6"
    elif ">" in expr:
        left, right = expr.split(">")
        return f"({left}) - ({right}) - 1e-6"
    else:
        raise ValueError(f"Invalid constraint format: {expr}")


def solve_from_initial_point(x0, cost_expr, var_names, constraint_exprs, bounds):
    cost_values = []

    def cost_fn(x):
        local_vars = dict(zip(var_names, x))
        val = eval(cost_expr, {}, local_vars)
        cost_values.append(val)
        return val

    cons = []
    for expr in constraint_exprs:
        if expr.strip():
            try:
                transformed_expr = parse_constraint_expression(expr.strip())
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, e=transformed_expr: eval(e, {}, dict(zip(var_names, x)))
                })
            except Exception as e:
                raise ValueError(f"Failed to parse constraint: {expr}\n{e}")
    print(f"Initial point: {x0}, Cost function: {cost_expr}, Constraints: {constraint_exprs}")
    result = minimize(cost_fn, x0, method='SLSQP', bounds=bounds, constraints=cons)
    result.cost_trace = cost_values
    return result


class OptimizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parallel Optimization with Multiple Initial Points")
        self.setGeometry(100, 100, 800, 700)

        self.state_input = QLineEdit()
        self.range_input = QTextEdit()
        self.cost_input = QLineEdit()
        self.constraints_input = QTextEdit()
        self.num_init_input = QLineEdit("5")
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)

        self.plot_canvas = FigureCanvas(plt.figure())

        self.solve_button = QPushButton("Solve")
        self.solve_button.clicked.connect(self.solve_optimization)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("State Variables (comma-separated, e.g., x, y):"))
        layout.addWidget(self.state_input)

        layout.addWidget(QLabel("Ranges (e.g., x: -5, 5):"))
        layout.addWidget(self.range_input)

        layout.addWidget(QLabel("Cost Function (e.g., x**2 + y**2):"))
        layout.addWidget(self.cost_input)

        layout.addWidget(QLabel("Constraints (comma-separated, e.g., x >= 0, x + y <= 5):"))
        layout.addWidget(self.constraints_input)

        layout.addWidget(QLabel("Number of Initial Points:"))
        layout.addWidget(self.num_init_input)

        layout.addWidget(self.solve_button)
        layout.addWidget(QLabel("Optimization Result:"))
        layout.addWidget(self.result_output)
        layout.addWidget(QLabel("Best Cost Trace:"))
        layout.addWidget(self.plot_canvas)

        self.setLayout(layout)

    def solve_optimization(self):
        try:
            var_names = [v.strip() for v in self.state_input.text().split(',')]
            n_vars = len(var_names)

            # Parse variable bounds
            bounds = []
            range_dict = {}
            for line in self.range_input.toPlainText().splitlines():
                if ':' not in line:
                    continue
                var, rng = line.split(':')
                low, high = map(float, rng.split(','))
                var = var.strip()
                range_dict[var] = (low, high)
                bounds.append((low, high))

            if set(var_names) != set(range_dict.keys()):
                raise ValueError("Mismatch between variables and defined ranges.")

            cost_expr = self.cost_input.text()
            constraints = self.constraints_input.toPlainText().split(',')

            num_init = int(self.num_init_input.text())
            initial_points = [
                np.array([np.random.uniform(*range_dict[v]) for v in var_names])
                for _ in range(num_init)
            ]

            self.result_output.setText("Running optimization...\n")
            results = []

            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(solve_from_initial_point, x0, cost_expr, var_names, constraints, bounds)
                    for x0 in initial_points
                ]
                for i, future in enumerate(as_completed(futures)):
                    try:
                        res = future.result()
                        if res.success:
                            results.append(res)
                            self.result_output.append(
                                f"[Init {i+1}] Success: Cost = {res.fun:.4f}, X = {res.x}"
                            )
                        else:
                            self.result_output.append(f"[Init {i+1}] Optimization failed.")
                    except Exception as e:
                        self.result_output.append(f"[Init {i+1}] Error: {str(e)}")

            if not results:
                self.result_output.append("No successful optimizations.")
                return

            best_result = min(results, key=lambda r: r.fun)
            self.result_output.append("\nBest Solution:")
            for name, val in zip(var_names, best_result.x):
                self.result_output.append(f"{name} = {val:.4f}")
            self.result_output.append(f"Minimum Cost: {best_result.fun:.4f}")

            self.plot_cost_trace(best_result.cost_trace)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_cost_trace(self, cost_trace):
        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)
        ax.plot(cost_trace, marker='o')
        ax.set_title("Best Cost Over Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.grid(True)
        self.plot_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OptimizationApp()
    window.show()
    sys.exit(app.exec_())
