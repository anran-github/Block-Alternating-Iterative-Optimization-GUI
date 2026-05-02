import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QTextEdit,
    QPushButton, QMessageBox, QHBoxLayout,QFileDialog,QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize
import sympy as sp
from sympy.calculus.util import minimum
from time import time
import json
from joblib import Parallel, delayed
from scipy.stats import qmc  # for LHS


# Create a safe evaluation environment
SAFE_ENV = {
    "np": np,             # allow np namespace if user writes np.sin
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "log": np.log,        # natural log
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e
}
SAFE_ENV.update({
    "trace": np.trace,
    "norm": np.linalg.norm
})

problem_data = {
            "xt": np.array([[1.79393],[29.9398]]),
            "xbar": np.array([[3.06228],[50]]),
            "ubar": 0,
            # "A": np.array([[0.9946, -0.002],[3.9583,0.6127]]),
            # "B": np.array([[0.0021],[-0.0135]]),
            "A": np.array([[0.9999, -0.0002],[0.4881,0.9528]]),
            "B": np.array([[0.0002],[-0.0022]]),
            "Qx": np.array([[0.01,0],[0,1000]]),
            "Qu": np.array([[0.0001]]),
            "theta": 0.001
        }


def evaluate_expression(expr, x, problem_data):
    u = x[0]
    P = np.array([
        [x[1], x[2]],
        [x[2], x[3]]
    ])

    xt = problem_data["xt"]
    xbar = problem_data["xbar"]
    ubar = problem_data["ubar"]
    A = problem_data["A"]
    B = problem_data["B"]

    x_next = A @ xt + B * u

    env = {
        **SAFE_ENV,
        "x": x,
        "u": u,
        "P": P,
        "xt": xt,
        "x_next": x_next,
        "xbar": xbar,
        "ubar": ubar,
        "A": A,
        "B": B,
        "Qx": problem_data["Qx"],
        "Qu": problem_data["Qu"],
        "theta": problem_data["theta"]
    }

    return eval(expr, {"__builtins__": {}}, env)


def unpack_vars(x, data):
    u = x[0]
    P = np.array([
        [x[1], x[2]],
        [x[2], x[3]]
    ])
    
    xt = data["xt"]       # x(t)
    xbar = data["xbar"]   # reference
    ubar = data["ubar"]

    A = data["A"]
    B = data["B"]

    x_next = A @ xt + B * u

    return u, P, xt, x_next, xbar, ubar, A, B

def generate_best_grid_points(bounds, cost_expr, constraint_exprs, S, problem_data, max_candidates=2000):
    import numpy as np
    from scipy.stats import qmc

    n = len(bounds)

    # -------- Helper: build structured variables (vectorized) --------
    def build_env_batch(X):
        # X: (N, n)
        N = X.shape[0]

        u = X[:, 0]

        # symmetric 2x2 P
        P = np.zeros((N, 2, 2))
        P[:, 0, 0] = X[:, 1]
        P[:, 0, 1] = X[:, 2]
        P[:, 1, 0] = X[:, 2]
        P[:, 1, 1] = X[:, 3]

        xt = problem_data["xt"]
        xbar = problem_data["xbar"]
        ubar = problem_data["ubar"]
        A = problem_data["A"]
        B = problem_data["B"]

        # x_next batch
        x_next = (A @ xt).reshape(1, -1) + (B @ u.reshape(1, -1)).T

        return u, P, xt, x_next, xbar, ubar, A, B

    # -------- Constraint evaluation (vectorized for known structure) --------
    def constraint_mask(X):
        mask = np.ones(X.shape[0], dtype=bool)

        # PSD constraint (Sylvester)
        mask &= X[:, 1] >= 1e-8
        mask &= (X[:, 1] * X[:, 3] - X[:, 2]**2) >= 1e-8

        # Lyapunov constraint
        u, P, xt, x_next, xbar, ubar, A, B = build_env_batch(X)

        diff_next = x_next - np.tile(xbar.T, (x_next.shape[0], 1))  # broadcast to (N, 2)
        diff_now = (xt - xbar).ravel()

        V_next = np.einsum('bi,bij,bj->b', diff_next, P, diff_next)
        # diff_now is constant across the batch, so use the 1-D vector directly.
        V_now  = np.einsum('i,bij,j->b', diff_now, P, diff_now)

        theta = problem_data["theta"]
        mask &= (V_next - V_now + theta * np.linalg.norm(diff_now)**2) <= 0

        return mask

    # -------- Cost evaluation (vectorized) --------
    def cost_batch(X):
        u, P, xt, x_next, xbar, ubar, A, B = build_env_batch(X)

        Qx = problem_data["Qx"]
        Qu = problem_data["Qu"]

        diff_next = x_next - xbar.T
        diff_now = (xt - xbar).ravel()

        term1 = np.einsum('bi,ij,bj->b', diff_next, Qx, diff_next)
        term2 = (u - ubar)**2 * Qu
        term3 = np.einsum('i,bij,j->b', diff_now, P, diff_now)
        term4 = np.einsum('bij,bij->b', P, P)

        return term1 + term2 + term3 + term4

    # -------- Sampling loop (vectorized) --------
    valid_points = []

    sampler = qmc.LatinHypercube(d=n)

    attempts = 0
    max_attempts = 20

    while len(valid_points) < S and attempts < max_attempts:
        lhs_sample = sampler.random(n=max_candidates)
        X = qmc.scale(lhs_sample,
                      [low for (low, high) in bounds],
                      [high for (low, high) in bounds])

        mask = constraint_mask(X)

        if np.any(mask):
            X_feasible = X[mask]
            costs = cost_batch(X_feasible)

            for c, pt in zip(costs[0], X_feasible):
                valid_points.append((c, pt))

        attempts += 1

    if len(valid_points) == 0:
        raise ValueError("No feasible points found. Try relaxing constraints or bounds.")

    # sort by cost
    valid_points.sort(key=lambda t: t[0])

    return [np.round(p, 5) for _, p in valid_points[:S]]

def check_convexity(cost_expr, bound, constraints):


    # Define function and variable    
    xi = sp.symbols('xi')
    cost_expr = cost_expr.replace("x[0]", "xi")  # Replace x[0] with x for sympy compatibility
    cost_expr = sp.sympify(cost_expr,evaluate=False)

    f2 = cost_expr.diff(xi, 2)  # Second derivative


    # Start with bound interval
    feasible_set = sp.Interval(bound[0][0], bound[0][1])

    # Apply constraints
    for c in constraints:
        if isinstance(c, str):
            c = c.replace("x[0]", "xi")
            c = sp.sympify(c)
        if c.is_Relational:
            sol_set = sp.solve_univariate_inequality(c, xi, relational=False)
            feasible_set = feasible_set.intersect(sol_set)

    # Check symbolically
    # try:
    symbolic_check = sp.reduce_inequalities(
        [f2 >= 0, xi >= feasible_set.inf, xi <= feasible_set.sup]
    )
    if symbolic_check == True:
        return True
    # except Exception:
    #     pass

    # Fallback numerical sampling
    lower, upper = float(feasible_set.inf), float(feasible_set.sup)
    sample_points = np.linspace(lower, upper, 500)
    is_convex_num = all(f2.subs(xi, val) >= 0 for val in sample_points)

    return is_convex_num


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


def solve_from_initial_point(x0, cost_expr, constraint_exprs, bounds, mode="SLSQP"):
    cost_values = []

    def cost_fn(x):
        val = evaluate_expression(cost_expr, x, problem_data)
        cost_values.append(float(val))
        return float(val)

    cons = []
    for expr in constraint_exprs:
        s = expr.strip()
        if not s:
            continue
        transformed = parse_constraint_expression(s)

        def make_fun(trans_expr):
            return lambda x: float(evaluate_expression(trans_expr, x, problem_data))

        cons.append({"type": "ineq", "fun": make_fun(transformed)})

    if mode == "Block-Alternating Iter":
        result = minimize(cost_fn, x0, method='SLSQP', bounds=bounds, constraints=cons,options={'maxiter': 5})
    else:
        result = minimize(cost_fn, x0, method=mode, bounds=bounds, constraints=cons) #, options={'disp': True, 'maxiter': 1000})

    result.cost_trace = cost_values
    return result


def optimize_point_worker(init_pt_x, n, bounds, cost_expr, constraint_list, problem_data):
    pt = init_pt_x.copy()
    results = []

    for i in range(n):
        def lifted_cost(x1d):
            x_full = pt.copy()
            x_full[i] = x1d[0]
            return evaluate_expression(cost_expr, x_full, problem_data)

        def lifted_constraints(x1d):
            x_full = pt.copy()
            x_full[i] = x1d[0]
            return [
                float(evaluate_expression(parse_constraint_expression(c), x_full, problem_data))
                for c in constraint_list
            ]

        cons = [{"type": "ineq", "fun": lambda x1d, k=k: lifted_constraints(x1d)[k]}
                for k in range(len(constraint_list))]

        res = minimize(
            lifted_cost,
            [pt[i]],
            bounds=[bounds[i]],
            constraints=cons,
            method="SLSQP",
            options={'maxiter': 5}
        )

        if res.success:
            pt[i] = float(res.x[0])
            results.append(res)

    return pt, results


class OptimizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Constrained Optimization Platform")
        self.setGeometry(100, 100, 900, 720)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Number of variables (n):"))
        self.num_vars_input = QLineEdit("4")
        top_row.addWidget(self.num_vars_input)

        top_row.addWidget(QLabel("Number of initial points:"))
        self.num_init_input = QLineEdit("8")
        top_row.addWidget(self.num_init_input)

        self.bounds_input = QTextEdit()
        # self.bounds_input.setPlaceholderText(
        #     "Examples:\n"
        #     "x1: -5, 5\n"
        #     "x2: 0, 10\n"
        #     "x3: 1        # shorthand -> [-1,1]\n"
        #     "all: -5, 5\n"
        #     "x1-x3: -2, 2\n" 
        #     "You may separate multiple entries with ';' on the same line."
        # )
        self.bounds_input.setText(
            "x1: 0, 100\n"     # u
            "x2: 0.001, 1000\n"  # p11
            "x3: -5, 50\n"     # p12
            "x4: 0.001, 1000\n"  # p22     
        )

        self.problem_data = problem_data

        self.cost_input = QLineEdit()
        # self.cost_input.setPlaceholderText("e.g. (x1-1)**2 + (x2+2)**2 + 0.5*(x3)**2")
        self.cost_input.setText("(x_next - xbar).T @ Qx @ (x_next - xbar) + (u - ubar)**2 * Qu+ (xt - xbar).T @ P @ (xt - xbar)+ trace(P.T @ P)")

        self.constraints_input = QTextEdit()
        # self.constraints_input.setPlaceholderText(
        #     "Constraints, comma-separated or newline separated. Use x1, x2, ...\n"
        #     "Example:\n x1 >= 0, x2 >= -2, x1 + x2 <= 5"
        # )
        self.constraints_input.setText(
            "x2 >= 1e-6\n"
            "x2*x4 - x3**2 >= 1e-6\n"
            "(x_next - xbar).T @ P @ (x_next - xbar) - (xt - xbar).T @ P @ (xt - xbar)+ theta * norm(xt - xbar)**2 <= 0"
        )

        solve_mode = QHBoxLayout()
        solve_mode.addWidget(QLabel("Select Optimization Algorithm:"))
        self.algo_selector = QComboBox()
        self.algo_selector.addItems(["Block-Alternating Iter", "SLSQP", "COBYLA","trust-constr", "GA", "Particle Swarm"])
        solve_mode.addWidget(self.algo_selector)

        self.solve_button = QPushButton("Solve (parallel)")
        self.solve_button.clicked.connect(self.solve_optimization)

        solvers_solve = QHBoxLayout()
        solvers_solve.addLayout(solve_mode)
        solvers_solve.addWidget(self.solve_button)

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
        
        # layout.addWidget(self.algo_selector)
        layout.addLayout(solvers_solve)
        # layout.addWidget(self.solve_button)
        layout.addWidget(QLabel("Optimization Results:"))
        layout.addWidget(self.result_output)
        layout.addWidget(QLabel("Best Cost Trace:"))
        layout.addWidget(self.plot_canvas)


        # Create Export button
        self.export_button = QPushButton("Export plot")
        self.export_button.clicked.connect(self.export_plot)

        # Add it at bottom-right
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch(1)  # push button to right
        bottom_layout.addWidget(self.export_button)
        layout.addLayout(bottom_layout)        

        self.setLayout(layout)


        self.executor = ProcessPoolExecutor()
        self.Parallel_Computation = Parallel(n_jobs=-1)


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
        


    def _solve_with_Multi_SCD(self,n, S, bounds, cost_expr, constraint_list, tol=1e-4, max_iters=50):
        self.result_output.clear()
        self.result_output.append("Running Block-Alternating Iteration Optimization...\n")


        n = len(bounds)

        # Latin Hypercube initial points
        t_start = time()
        initial_points = generate_best_grid_points(bounds, cost_expr, constraint_list, S, problem_data=self.problem_data)

        cost_results = []
        all_results_buffer = []
        diff = tol + 1
        iter_count = 0

        while diff > tol and iter_count < max_iters:


            # run all points in parallel
            parallel_results = self.Parallel_Computation(
                                delayed(optimize_point_worker)(
                                    init_pt_x, n, bounds, cost_expr, constraint_list, self.problem_data
                                )
                                for init_pt_x in initial_points)

            # unpack updated points and results
            updated_points, all_results = zip(*parallel_results)
            initial_points = np.array(updated_points)

            
            flat_results = [r for res_list in all_results for r in res_list if r.success]
            
            all_results_buffer.append(all_results)


            if flat_results:
                best_index = np.argmin([r.fun for r in flat_results])
                current_best = flat_results[best_index].fun
                if cost_results:
                    diff = abs(cost_results[-1] - current_best)
                cost_results.append(current_best)

            iter_count += 1


        t_end = time()
        min_cost = []
        for res_idx,res_list in enumerate(all_results):
            if len(res_list) == 0:
                min_cost.append(float('inf'))  # no valid results
                continue
            # select the min cost from each initial point's results
            min_cost.append(min(r.fun for r in res_list if r.success))

        lowest_idx = np.argmin(min_cost)

        best = min(flat_results, key=lambda r: r.fun)
        # selected_idx = flat_results.index(best)
        best_result = initial_points[lowest_idx]

        self.result_output.append("\n=== Best Solution ===")
        for i, val in enumerate(best_result):
            self.result_output.append(f"x{i+1} = {float(val):.8g}")
        self.result_output.append(f"Minimum cost: {best.fun:.12g}")
        self.result_output.append(f"Total time cost: {t_end - t_start:.4f} seconds")

        # best_cost_trace = [x[selected_idx] for x in cost_results]
        cost_results_plot = []
        for init_pt_x in all_results_buffer:
            if len(init_pt_x) == len(initial_points):
                cost_results_plot.extend([x.fun for x in init_pt_x[lowest_idx]])
        
        self._plot_cost_trace(cost_results_plot,method="Multi_SCD")

        trace_len = len(best.cost_trace) if hasattr(best, "cost_trace") else "N/A"

        print(f"Best solution: {best.x}, cost: {best.fun:.6g}, evaluations: {trace_len}")


    def _solve_with_GA(self, n, num_init, bounds, cost_expr, constraint_list,
                    generations=200, crossover_rate=0.8, mutation_rate=0.2):
        """
        Genetic Algorithm solver using initial points as seed population.
        Matrix-support added (no change to GA logic).
        """
        self.result_output.clear()
        self.result_output.append("Running Genetic Algorithm Optimization...\n")

        t_start = time()

        # 1. Initial population
        initial_points = generate_best_grid_points(
            bounds, cost_expr, constraint_list, num_init,
            problem_data=self.problem_data
        )

        pop_size = num_init
        population = [np.array(p, dtype=float) for p in initial_points]

        while len(population) < pop_size:
            rand_point = np.array([np.random.uniform(low, high) for (low, high) in bounds], dtype=float)
            population.append(rand_point)

        population = np.array(population, dtype=object)  # allow matrix objects

        # --- Fitness evaluation (matrix-safe) ---
        def evaluate(ind):
            x = np.asarray(ind)

            # build same variable environment as PSO
            try:
                u, P, xt, x_next, xbar, ubar, A, B = unpack_vars(x, self.problem_data)
            except Exception:
                # fallback if unpack fails (keeps GA robust)
                u = P = xt = x_next = xbar = ubar = A = B = None

            env = {
                **SAFE_ENV,
                "x": x,
                "u": u,
                "P": P,
                "xt": xt,
                "x_next": x_next,
                "xbar": xbar,
                "ubar": ubar,
                "A": A,
                "B": B,
                "Qx": self.problem_data.get("Qx", None),
                "Qu": self.problem_data.get("Qu", None),
                "theta": self.problem_data.get("theta", None),
            }

            # constraint check (now supports x_next etc.)
            try:
                feasible = all(
                    eval(parse_constraint_expression(c),
                        {"__builtins__": {}},
                        env) >= 0
                    for c in constraint_list
                )
            except NameError:
                # if any variable still missing → treat as infeasible safely
                return 1e10

            if feasible:
                return float(np.asarray(eval(cost_expr, {"__builtins__": {}}, env)).squeeze())
            else:
                return 1e10

        cost_trace = []
        best_individual = None
        best_cost = float("inf")

        # 2. GA loop
        for gen in range(generations):
            fitness = np.array([evaluate(ind) for ind in population])

            gen_best_idx = np.argmin(fitness)
            gen_best_cost = fitness[gen_best_idx]

            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_individual = population[gen_best_idx].copy()

            cost_trace.append(best_cost)
            self.result_output.append(
                        f"Generation {gen}: best cost = {float(np.asarray(best_cost).squeeze()):.6g}"
                    )

            # 3. Selection (tournament)
            selected = []
            for _ in range(pop_size):
                i, j = np.random.randint(0, pop_size, 2)
                selected.append(population[i] if fitness[i] < fitness[j] else population[j])
            selected = np.array(selected, dtype=object)

            # 4. Crossover (matrix-safe)
            offspring = []
            for i in range(0, pop_size, 2):
                p1, p2 = selected[i], selected[(i + 1) % pop_size]

                if np.random.rand() < crossover_rate:
                    alpha = np.random.rand()
                    child1 = alpha * p1 + (1 - alpha) * p2
                    child2 = alpha * p2 + (1 - alpha) * p1
                else:
                    child1, child2 = p1.copy(), p2.copy()

                offspring.extend([child1, child2])

            offspring = np.array(offspring, dtype=object)

            # 5. Mutation (matrix-safe)
            for ind in offspring:
                if np.random.rand() < mutation_rate:
                    k = np.random.randint(n)
                    ind[k] = np.random.uniform(bounds[k][0], bounds[k][1])

            # 6. Elitism
            worst_idx = np.argmax([evaluate(ind) for ind in offspring])
            offspring[worst_idx] = best_individual.copy()

            population = offspring

        # Final results
        t_end = time()

        self.result_output.append("\n=== Best Solution (GA) ===")
        for i, val in enumerate(best_individual):
            self.result_output.append(f"x{i+1} = {float(val):.8g}")

        self.result_output.append(f"Minimum cost: {best_cost:.12g}")
        self.result_output.append(f"Total time cost: {t_end - t_start:.4f} seconds")

        self._plot_cost_trace(cost_trace, method="GA")

    def _solve_with_PSO(self, n, num_init, bounds, cost_expr, constraint_list,
                        max_iter=200, w=0.7, c1=1.5, c2=1.5):
        """
        Particle Swarm Optimization (matrix-support added, logic unchanged)
        """
        swarm_size = num_init
        self.result_output.clear()
        self.result_output.append("Running Particle Swarm Optimization...\n")

        t_start = time()

        # 1. Initial particles
        initial_points = generate_best_grid_points(
            bounds, cost_expr, constraint_list, num_init,
            problem_data=self.problem_data
        )

        particles = [np.array(p, dtype=float) for p in initial_points]

        while len(particles) < swarm_size:
            rand_point = np.array([np.random.uniform(low, high) for (low, high) in bounds], dtype=float)
            particles.append(rand_point)

        particles = np.array(particles, dtype=object)

        velocities = np.zeros_like(particles, dtype=object)

        # --- Matrix-safe evaluation ---
        def evaluate(x):
            x = np.asarray(x)

            # ---- SAFE unpack (NEVER leave None variables) ----
            try:
                u, P, xt, x_next, xbar, ubar, A, B = unpack_vars(x, self.problem_data)
            except Exception:
                dim = len(x)
                u = P = xt = x_next = xbar = ubar = A = B = np.zeros(dim)

            # ---- FORCE SAFE NUMERIC TYPES ----
            x = np.asarray(x, dtype=float)
            u = np.asarray(u, dtype=float)
            P = np.asarray(P, dtype=float)
            xt = np.asarray(xt, dtype=float)
            x_next = np.asarray(x_next, dtype=float)
            xbar = np.asarray(xbar, dtype=float)
            ubar = np.asarray(ubar, dtype=float)
            A = np.asarray(A, dtype=float)
            B = np.asarray(B, dtype=float)

            env = {
                **SAFE_ENV,
                "x": x,
                "u": u,
                "P": P,
                "xt": xt,
                "x_next": x_next,
                "xbar": xbar,
                "ubar": ubar,
                "A": A,
                "B": B,
                "Qx": self.problem_data.get("Qx", 0),
                "Qu": self.problem_data.get("Qu", 0),
                "theta": self.problem_data.get("theta", 0),
            }

            # ---- constraint check (fully safe now) ----
            try:
                feasible = True
                for c in constraint_list:
                    expr = parse_constraint_expression(c)
                    val = eval(expr, {"__builtins__": {}}, env)

                    if np.any(np.asarray(val) < 0):
                        feasible = False
                        break

            except Exception:
                return 1e10

            if not feasible:
                return 1e10

            # ---- cost (force scalar) ----
            try:
                val = eval(cost_expr, {"__builtins__": {}}, env)
                return float(np.asarray(val).squeeze())
            except Exception:
                return 1e10
            
        # 2. Initialize pbest / gbest
        pbest = particles.copy()
        pbest_cost = np.array([evaluate(p) for p in pbest])

        gbest_idx = np.argmin(pbest_cost)
        gbest = pbest[gbest_idx].copy()
        gbest_cost = pbest_cost[gbest_idx]

        cost_trace = [gbest_cost]

        # 3. PSO loop
        for iter in range(max_iter):
            for i in range(swarm_size):

                r1, r2 = np.random.rand(n), np.random.rand(n)

                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (pbest[i] - particles[i]) +
                    c2 * r2 * (gbest - particles[i])
                )

                particles[i] = particles[i] + velocities[i]

                # bounds clipping (matrix-safe)
                for d in range(n):
                    low, high = bounds[d]
                    particles[i][d] = np.clip(particles[i][d], low, high)

                cost = evaluate(particles[i])

                if cost < pbest_cost[i]:
                    pbest[i] = particles[i].copy()
                    pbest_cost[i] = cost

                    if cost < gbest_cost:
                        gbest = particles[i].copy()
                        gbest_cost = cost

            cost_trace.append(gbest_cost)
            self.result_output.append(f"Iteration {iter+1}: best cost = {gbest_cost:.6g}")

        t_end = time()

        self.result_output.append("\n=== Best Solution (PSO) ===")
        for i, val in enumerate(gbest):
            self.result_output.append(f"x{i+1} = {float(val):.8g}")

        self.result_output.append(f"Minimum cost: {gbest_cost:.12g}")
        self.result_output.append(f"Total time cost: {t_end - t_start:.4f} seconds")

        self._plot_cost_trace(cost_trace, method="PSO")



    def _solve_with_other_inernal_opt(self, n, num_init, bounds, cost_expr, constraint_list,mode):
        self.result_output.clear()
        self.result_output.append(f"Running {mode} Optimization...\n")

        t_start = time()
        # generate initial points (just take first point from grid for simplicity)
        initial_points = generate_best_grid_points(bounds, cost_expr, constraint_list, num_init, problem_data=self.problem_data)

        x0 = initial_points[0]

        res = solve_from_initial_point(x0, cost_expr, constraint_list, bounds,mode)
        t_end = time()
        
        if res.success:
            self.result_output.append(f"\n=== Best Solution ({mode}) ===")
            for i, val in enumerate(res.x):
                self.result_output.append(f"x{i+1} = {float(val):.8g}")
            self.result_output.append(f"Minimum cost: {res.fun:.12g}")
            self.result_output.append(f"Total time cost: {t_end - t_start:.4f} seconds")
            self._plot_cost_trace(res.cost_trace,method=mode)
        else:
            self.result_output.append(f"Optimization failed: {res.message}")


    def solve_optimization(self):
        # try:
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


        # select optimization algorithm
        algo = self.algo_selector.currentText()
        if algo == "SLSQP" or algo == "COBYLA" or algo == "trust-constr":
            # use SLSQP or COBYLA or trust-constr
            self._solve_with_other_inernal_opt(n, num_init, bounds, cost_expr, constraint_list,algo)
        elif algo == "Block-Alternating Iter":
            self._solve_with_Multi_SCD(n, num_init, bounds, cost_expr, constraint_list)
        elif algo == "GA":
            self._solve_with_GA(n, num_init, bounds, cost_expr, constraint_list)
        elif algo == "Particle Swarm":
            self._solve_with_PSO(n, num_init, bounds, cost_expr, constraint_list)
    
        # except Exception as e:
        #     QMessageBox.critical(self, "Error", str(e))

    def _format_solution(self, arr):
        return "[" + ", ".join(f"x{i+1}={v:.6g}" for i, v in enumerate(arr)) + "]"


    def _plot_cost_trace(self, trace,method="Multi_SCD"):
        self.plot_data = {"method": method, "results": trace}
        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)
        ax.plot(trace, marker='o')
        ax.set_title("Cost evaluations (best run)")
        ax.set_xlabel("Function evaluation index")
        ax.set_ylabel("Cost")
        ax.grid(True)
        self.plot_canvas.draw()


    # Function to handle export
    def export_plot(self):
        if not hasattr(self, "plot_data") or not self.plot_data:
            QMessageBox.warning(self, "No Data", "No plot data available to export.")
            return

        # Example: assume you store results in self.plot_data like:
        # self.plot_data = {"method": "Multi_SCD", "results": [(x1,y1), (x2,y2), ...]}
        method = self.plot_data.get("method", "unknown")
        filename = f"{method}_results.json"

        # Ask user where to save
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Plot Data", filename, "JSON Files (*.json)")
        if not filepath:
            return

        try:
            with open(filepath, "w") as f:
                json.dump(self.plot_data, f, indent=4)

            QMessageBox.information(self, "Export Successful", f"File saved: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OptimizationApp()
    win.show()
    sys.exit(app.exec_())
