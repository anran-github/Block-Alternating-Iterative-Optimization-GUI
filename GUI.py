import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class OptimizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coordinate Descent Optimizer")
        self.setGeometry(200, 200, 800, 800)

        self.iter_values = []

        # Widgets
        self.state_input = QLineEdit()
        self.range_input = QTextEdit()
        self.cost_input = QLineEdit()
        self.constraints_input = QTextEdit()
        self.result_output = QLabel()

        self.plot_canvas = FigureCanvas(plt.figure())

        self.solve_button = QPushButton("Solve")
        self.solve_button.clicked.connect(self.solve_optimization)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("State Variables (comma-separated):"))
        layout.addWidget(self.state_input)

        layout.addWidget(QLabel("Ranges (e.g., x: -5, 5\\ny: 0, 10):"))
        layout.addWidget(self.range_input)

        layout.addWidget(QLabel("Cost Function (e.g., x**2 + y**2):"))
        layout.addWidget(self.cost_input)

        layout.addWidget(QLabel("Constraints (comma-separated, e.g., x >= 0, y + x <= 5):"))
        layout.addWidget(self.constraints_input)

        layout.addWidget(self.solve_button)
        layout.addWidget(self.result_output)
        layout.addWidget(self.plot_canvas)

        self.setLayout(layout)

    def solve_optimization(self):
        # Parse variables
        try:
            var_names = [v.strip() for v in self.state_input.text().split(',')]
            ranges = {}
            for line in self.range_input.toPlainText().splitlines():
                var, rng = line.split(':')
                low, high = map(float, rng.split(','))
                ranges[var.strip()] = (low, high)

            cost_expr = self.cost_input.text()
            constraints = self.constraints_input.toPlainText().split(',')

            def evaluate_cost(values):
                local_vars = dict(zip(var_names, values))
                return eval(cost_expr, {}, local_vars)

            def satisfies_constraints(values):
                local_vars = dict(zip(var_names, values))
                for cons in constraints:
                    if cons.strip() == '':
                        continue
                    if not eval(cons.strip(), {}, local_vars):
                        return False
                return True

            # Coordinate Descent
            x = np.array([(ranges[v][0] + ranges[v][1]) / 2 for v in var_names])
            step_size = 0.1
            max_iter = 100
            self.iter_values = []

            for i in range(max_iter):
                for j in range(len(var_names)):
                    best_val = x[j]
                    best_cost = evaluate_cost(x) if satisfies_constraints(x) else float('inf')
                    for delta in np.linspace(-step_size, step_size, 10):
                        new_x = x.copy()
                        new_x[j] += delta
                        # Enforce bounds
                        new_x[j] = np.clip(new_x[j], *ranges[var_names[j]])
                        if satisfies_constraints(new_x):
                            cost = evaluate_cost(new_x)
                            if cost < best_cost:
                                best_val = new_x[j]
                                best_cost = cost
                    x[j] = best_val
                self.iter_values.append(evaluate_cost(x) if satisfies_constraints(x) else np.nan)

            # Show result
            result_text = f"Optimal Variables:\n"
            for name, val in zip(var_names, x):
                result_text += f"{name} = {val:.4f}\n"
            result_text += f"Minimum Cost: {evaluate_cost(x):.4f}"
            self.result_output.setText(result_text)

            self.plot_iterations()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_iterations(self):
        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)
        ax.plot(self.iter_values, marker='o')
        ax.set_title("Objective Function Over Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.grid(True)
        self.plot_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OptimizationApp()
    window.show()
    sys.exit(app.exec_())
