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
# import re
# import numexpr as ne



# def generate_best_grid_points(bounds, cost_expr, constraint_exprs, S):
#     # Generate grid points for each variable
#     total_vars = len(bounds)
#     # assign number of points based on variable ranges:
#     num_pts = []
#     grids = []
#     for (low, high) in bounds:
#         if  (high - low) <= 10:
#             num_pts.append(50)
#         elif (high - low) > 10 and (high - low) <= 100:
#             num_pts.append(100)
#         elif (high - low) > 100 and (high - low) <= 500:
#             num_pts.append(500)
#         elif (high - low) > 500:
#             num_pts.append(1000)

#         grids.append(np.linspace(low, high, num_pts[-1]))
            

#     # grids = [np.linspace(low, high, step) for enumerate(low, high) in bounds]
#     mesh = np.meshgrid(*grids, indexing='xy')
#     candidates = np.vstack([m.flatten() for m in mesh]).T
#     print(candidates.shape, "candidates shape")
#     # Prepare constraint functions
#     constraint_funcs = []
#     for expr in constraint_exprs:
#         if expr.strip():
#             transformed = parse_constraint_expression(expr.strip())
#             constraint_funcs.append(lambda x, e=transformed: eval(e, {}, {"x": x}) >= 0)

#     valid_points = []
#     for point in candidates:
#         # Check constraints
#         if all(f(point) for f in constraint_funcs):
#             cost_val = eval(cost_expr, {}, {"x": point})
#             valid_points.append((cost_val, point))
    
#     # Sort by cost and take top S
#     valid_points.sort(key=lambda t: t[0])
#     return [np.round(p,5) for _, p in valid_points[:S]]


def generate_best_grid_points(bounds, cost_expr, constraint_exprs, S, max_candidates=100):
    n = len(bounds)

    grids = []
    for (low, high) in bounds:
        grids.append(np.linspace(low, high, 50))
            

    # grids = [np.linspace(low, high, step) for enumerate(low, high) in bounds]
    mesh = np.meshgrid(*grids, indexing='xy')
    candidates_uniform = np.vstack([m.flatten() for m in mesh]).T

    valid_points = []
    constraint_funcs = [lambda x, e=parse_constraint_expression(c): eval(e, {}, {"x": x}) >= 0 for c in constraint_exprs if c.strip()]
    
    for pt in candidates_uniform:
        if all(f(pt) for f in constraint_funcs):
            cost_val = eval(cost_expr, {}, {"x": pt})
            valid_points.append((cost_val, pt))

    # add more feasible points if not enough
    while len(valid_points) < S:
        candidates = np.array([
            [np.random.uniform(low, high) for (low, high) in bounds]
            for _ in range(max_candidates)
        ])
        # evaluate cost only on feasible
        
        for pt in candidates:
            if all(f(pt) for f in constraint_funcs):
                cost_val = eval(cost_expr, {}, {"x": pt})
                valid_points.append((cost_val, pt))
                
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
    try:
        symbolic_check = sp.reduce_inequalities(
            [f2 >= 0, xi >= feasible_set.inf, xi <= feasible_set.sup]
        )
        if symbolic_check == True:
            return True
    except Exception:
        pass

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

    if mode == "Multi-Start CD":
        result = minimize(cost_fn, x0, method='SLSQP', bounds=bounds, constraints=cons,options={'maxiter': 5})
    else:
        result = minimize(cost_fn, x0, method=mode, bounds=bounds, constraints=cons) #, options={'disp': True, 'maxiter': 1000})

    result.cost_trace = cost_values
    return result


class OptimizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-inital Coordinate Descent Optimization")
        self.setGeometry(100, 100, 900, 720)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Number of variables (n):"))
        self.num_vars_input = QLineEdit("3")
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
            "x1: -3, 3\n"
            "x2: 0, 10\n"
            "x3: -1, 1 "      
        )


        self.cost_input = QLineEdit()
        # self.cost_input.setPlaceholderText("e.g. (x1-1)**2 + (x2+2)**2 + 0.5*(x3)**2")
        self.cost_input.setText("(x1-x2)**2 + (1/x2+2)**2 + 0.5*(x3)**2")

        self.constraints_input = QTextEdit()
        # self.constraints_input.setPlaceholderText(
        #     "Constraints, comma-separated or newline separated. Use x1, x2, ...\n"
        #     "Example:\n x1 >= 0, x2 >= -2, x1 + x2 <= 5"
        # )
        self.constraints_input.setText(
            "x1 >= 0\n"
            "x2 >= 2\n"
            "x1 + x2 <= 5\n"
            "x3*x1 >= 2\n"
        )

        solve_mode = QHBoxLayout()
        solve_mode.addWidget(QLabel("Select Optimization Algorithm:"))
        self.algo_selector = QComboBox()
        self.algo_selector.addItems(["Multi-Start CD", "SLSQP", "COBYLA","trust-constr", "GA", "Particle Swarm"])
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
        


    def _solve_with_Multi_SCD(self, n, num_init, bounds, cost_expr, constraint_list):

        self.result_output.append("Running Multi-Starting Point CD Optimization...\n")

        t_start = time()
        initial_points = generate_best_grid_points(bounds, cost_expr, constraint_list, num_init)

        # solve the optimization problem for the i-th variable
        self.result_output.clear()
        

        cnt,i = 0,0
        tolerance = 1e-6
        diff = 1.0
        cost_results = []
        # start Coordinate Descent with multiple initial points
        while diff > tolerance and cnt < 100:
            
            self.result_output.append(f"Running Multi-Point CD iteraion {cnt}...\n")
            
            order = list(range(n))
            # np.random.shuffle(order)  # random order of variables
            for i in order:
                
                # if diff <= tolerance:
                #     break
                # set varialbes after i-th variable to init values
                # print('initial points:', initial_points)
                x0_c = [x[i] for x in initial_points]
                bounds_c = [bounds[i]]

                cost_expr_c_tmp = cost_expr
                cost_expr_c = []
                constraint_list_c_tmp = constraint_list.copy()
                constraint_list_c = []
                for init_pt_x in initial_points:
                    for j in range(n):
                        if j != i:
                            cost_expr_c_tmp = cost_expr_c_tmp.replace(f"x[{j}]", f"{init_pt_x[j]}")
                            constraint_list_c_tmp = [c.replace(f"x{j+1}", f"{init_pt_x[j]}") for c in constraint_list_c_tmp if f"x{i+1}" in c]
                            
                            
                            # replace xi to x1, now we only have one variable
                    cost_expr_c_tmp_new = cost_expr_c_tmp.replace(f"x[{i}]", f"x[0]")
                    
                    cost_expr_c.append(cost_expr_c_tmp_new)
                    constraint_list_c_tmp_new = [x.replace(f"x{i+1}", f"x[0]") for x in constraint_list_c_tmp if f"x{i+1}" in x]
                    constraint_list_c.append(constraint_list_c_tmp_new)


                # print(f"Initial points for variable {i+1}: {x0_c}")
                # print(f"Bounds for variable {i+1}: {bounds_c}")
                # print(f"Constraints for variable {i+1}: {constraint_list_c}")
                # print(f"Cost expr for variable {i+1}: {cost_expr_c}")
                # check convexity
                
                # WARNING: THIS CHECKING OPERATION IS EXPENSIVE, COMMENT OUT TO SPEED UP.
                # if cnt == 0:
                #     is_convex = check_convexity(cost_expr_c[0],bounds_c, constraint_list_c[0])
                #     if not is_convex:
                #         self.result_output.append("Cost function is not convex in the given constraints:")
                #         self.result_output.append(f"When processing variable x{i+1}")
                #         return

                results = []
                futures = [
                    self.executor.submit(solve_from_initial_point, x0, cost_expr_i, constraint_c, bounds_c,"Multi-Start CD")
                    for x0,cost_expr_i,constraint_c in zip(x0_c,cost_expr_c,constraint_list_c)
                ]
                for idx, future in enumerate(as_completed(futures), start=1):
                    try:
                        res = future.result()
                        if res.success:
                            
                            results.append(res)

                            # self.result_output.append(
                            #     f"[Init {idx}] Success: cost={res.fun:.6g}, x={self._format_solution(res.x)}"
                            # )
                        else:
                            self.result_output.append(f"[Init {idx}] Failed: {res.message}")
                            results.append(res)  # still collect results
                    except Exception as e:
                        self.result_output.append(f"[Init {idx}] Error: {str(e)}")

                if not results:
                    self.result_output.append("\nNo successful runs found.")
                    return

                

                # build new init points for next iteration
                for k, res in enumerate(results):
                    if res.success:
                        new_x = res.x.copy()
                        initial_points[k][i] = new_x  # update i-th variable
                    else:
                        self.result_output.append(f"Run {k+1} failed, keeping old initial point for variable {i+1}")


                # calculate the difference between the best cost and the previous cost
                last_cost = [res.fun for res in results]
                if cnt == 0 and i == 0:
                    cost_results.append(last_cost)
                else:
                    previous_cost_idx = results.index(min(results, key=lambda r: r.fun))
                    if results[previous_cost_idx].success:
                        diff = abs(cost_results[-1][previous_cost_idx] - results[previous_cost_idx].fun)

                    cost_results.append(last_cost)
                    
            cnt += 1


        t_end = time()
        best = min(results, key=lambda r: r.fun)
        selected_idx = results.index(best)
        best_result = initial_points[selected_idx]

        self.result_output.append("\n=== Best Solution ===")
        for i, val in enumerate(best_result):
            self.result_output.append(f"x{i+1} = {float(val):.8g}")
        self.result_output.append(f"Minimum cost: {best.fun:.12g}")
        self.result_output.append(f"Total time cost: {t_end - t_start:.4f} seconds")

        best_cost_trace = [x[selected_idx] for x in cost_results]
        self._plot_cost_trace(best_cost_trace,method="Multi_SCD")

        print(f"Best solution: {best.x}, cost: {best.fun:.6g}, evaluations: {len(best.cost_trace)}")


    def _solve_with_other_inernal_opt(self, n, num_init, bounds, cost_expr, constraint_list,mode):
        self.result_output.clear()
        self.result_output.append(f"Running {mode} Optimization...\n")

        t_start = time()
        # generate initial points (just take first point from grid for simplicity)
        initial_points = generate_best_grid_points(bounds, cost_expr, constraint_list, 1)

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


    def _solve_with_GA(self, n, num_init, bounds, cost_expr, constraint_list,
                    generations=200, crossover_rate=0.8, mutation_rate=0.2):
        """
        Genetic Algorithm solver using initial points as seed population.
        """
        self.result_output.clear()
        self.result_output.append("Running Genetic Algorithm Optimization...\n")

        # 1. Generate initial points (from user grid selection)
        t_start = time()
        initial_points = generate_best_grid_points(bounds, cost_expr, constraint_list, num_init)
        pop_size = num_init  # Use num_init as population size
        # Ensure we have enough for GA population (fill with random if needed)
        population = []
        for p in initial_points:
            population.append(np.array(p, dtype=float))
        while len(population) < pop_size:
            rand_point = np.array([np.random.uniform(low, high) for (low, high) in bounds])
            population.append(rand_point)
        population = np.array(population)

        # Helper: evaluate fitness with constraints
        def evaluate(ind):
            if all(eval(parse_constraint_expression(c), {}, {"x": ind}) >= 0 for c in constraint_list):
                return eval(cost_expr, {}, {"x": ind})
            else:
                return 1e10  # penalize infeasible

        cost_trace = []
        best_individual = None
        best_cost = float("inf")

        # 2. Run generations
        for gen in range(generations):
            fitness = np.array([evaluate(ind) for ind in population])

            # Track best
            gen_best_idx = np.argmin(fitness)
            gen_best_cost = fitness[gen_best_idx]
            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_individual = population[gen_best_idx].copy()

            cost_trace.append(best_cost)
            self.result_output.append(f"Generation {gen}: best cost = {best_cost:.6g}")

            # 3. Selection (tournament)
            selected = []
            for _ in range(pop_size):
                i, j = np.random.randint(0, pop_size, 2)
                selected.append(population[i] if fitness[i] < fitness[j] else population[j])
            selected = np.array(selected)

            # 4. Crossover
            offspring = []
            for i in range(0, pop_size, 2):
                p1, p2 = selected[i], selected[(i+1) % pop_size]
                if np.random.rand() < crossover_rate:
                    alpha = np.random.rand()
                    child1 = alpha * p1 + (1 - alpha) * p2
                    child2 = alpha * p2 + (1 - alpha) * p1
                else:
                    child1, child2 = p1.copy(), p2.copy()
                offspring.extend([child1, child2])

            offspring = np.array(offspring)

            # 5. Mutation
            for ind in offspring:
                if np.random.rand() < mutation_rate:
                    k = np.random.randint(n)
                    ind[k] = np.random.uniform(bounds[k][0], bounds[k][1])

            population = offspring

        # Final results
        t_end = time()
        self.result_output.append("\n=== Best Solution (GA) ===")
        for i, val in enumerate(best_individual):
            self.result_output.append(f"x{i+1} = {float(val):.8g}")
        self.result_output.append(f"Minimum cost: {best_cost:.12g}")
        self.result_output.append(f"Total time cost: {t_end - t_start:.4f} seconds")
        self._plot_cost_trace(cost_trace,method="GA")


    def _solve_with_PSO(self, n, num_init, bounds, cost_expr, constraint_list,
                         max_iter=200, w=0.7, c1=1.5, c2=1.5):
        """
        Particle Swarm Optimization
        w: inertia weight
        c1, c2: cognitive and social coefficients
        """
        swarm_size = num_init
        self.result_output.clear()
        self.result_output.append("Running Particle Swarm Optimization...\n")

        # Helper to evaluate cost with constraint penalty
        def evaluate(x):
            feasible = all(eval(parse_constraint_expression(c), {}, {"x": x}) >= 0 for c in constraint_list)
            if feasible:
                return eval(cost_expr, {}, {"x": x})
            else:
                return 1e10  # penalize infeasible

        t_start = time()

        # 1. Seed initial particles from grid
        initial_points = generate_best_grid_points(bounds, cost_expr, constraint_list, num_init)

        particles = []
        for p in initial_points:
            particles.append(np.array(p, dtype=float))
        while len(particles) < swarm_size:
            rand_point = np.array([np.random.uniform(low, high) for (low, high) in bounds])
            particles.append(rand_point)
        particles = np.array(particles)

        # 2. Initialize velocities
        velocities = np.zeros_like(particles)

        # 3. Initialize personal and global bests
        pbest = particles.copy()
        pbest_cost = np.array([evaluate(p) for p in pbest])

        gbest_idx = np.argmin(pbest_cost)
        gbest = pbest[gbest_idx].copy()
        gbest_cost = pbest_cost[gbest_idx]

        cost_trace = [gbest_cost]

        # 4. Main PSO loop
        for iter in range(max_iter):
            for i in range(swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(n), np.random.rand(n)
                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (pbest[i] - particles[i]) +
                    c2 * r2 * (gbest - particles[i])
                )

                # Update position with velocity
                particles[i] = particles[i] + velocities[i]

                # Enforce bounds
                for d in range(n):
                    low, high = bounds[d]
                    particles[i][d] = np.clip(particles[i][d], low, high)

                # Evaluate
                cost = evaluate(particles[i])

                # Update personal best
                if cost < pbest_cost[i]:
                    pbest[i] = particles[i].copy()
                    pbest_cost[i] = cost

                    # Update global best
                    if cost < gbest_cost:
                        gbest = particles[i].copy()
                        gbest_cost = cost

            cost_trace.append(gbest_cost)
            self.result_output.append(f"Iteration {iter+1}: best cost = {gbest_cost:.6g}")

        time_end = time()
        
        # Final results
        self.result_output.append("\n=== Best Solution (PSO) ===")
        for i, val in enumerate(gbest):
            self.result_output.append(f"x{i+1} = {float(val):.8g}")
        self.result_output.append(f"Minimum cost: {gbest_cost:.12g}")
        self.result_output.append(f"Total time cost: {time_end - t_start:.4f} seconds")

        self._plot_cost_trace(cost_trace,method="PSO")



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


            # select optimization algorithm
            algo = self.algo_selector.currentText()
            if algo == "SLSQP" or algo == "COBYLA" or algo == "trust-constr":
                # use SLSQP or COBYLA or trust-constr
                self._solve_with_other_inernal_opt(n, num_init, bounds, cost_expr, constraint_list,algo)
            elif algo == "Multi-Start CD":
                self._solve_with_Multi_SCD(n, num_init, bounds, cost_expr, constraint_list)
            elif algo == "GA":
                self._solve_with_GA(n, num_init, bounds, cost_expr, constraint_list)
            elif algo == "Particle Swarm":
                self._solve_with_PSO(n, num_init, bounds, cost_expr, constraint_list)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

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
