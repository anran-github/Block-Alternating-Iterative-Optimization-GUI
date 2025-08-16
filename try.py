"""
Genetic Algorithm for constrained, non-convex optimization (real-coded)

Features
- Bounds on variables
- Inequality (g(x) <= 0) and equality (h(x) = 0) constraints
- Deb's feasibility rules for selection (no penalty weight tuning!)
- SBX crossover + polynomial mutation (both standard in evolutionary comp.)
- Tournament selection, elitism, reproducible RNG seed
- Plug-in objective/constraints

Usage (see __main__ at bottom for a worked example)
- Define objective(x: np.ndarray) -> float
- Define constraints with two callables:
    ineqs(x) -> array-like of g_i(x) where each must be <= 0
    eqs(x)   -> array-like of h_j(x) where each must be == 0 (with tolerance)
- Instantiate GA and call run()

References
- Kalyanmoy Deb (2000). An efficient constraint handling method for genetic algorithms.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Optional, List
import numpy as np

ObjectiveFn = Callable[[np.ndarray], float]
IneqFn = Callable[[np.ndarray], Iterable[float]]
EqFn = Callable[[np.ndarray], Iterable[float]]

@dataclass
class GAConfig:
    pop_size: int = 200
    generations: int = 300
    tournament_k: int = 3
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1  # per-variable probability
    sbx_eta: float = 15.0
    mut_eta: float = 20.0
    elite_fraction: float = 0.02
    eq_tol: float = 1e-4  # equality tolerance
    seed: Optional[int] = 42

class ConstrainedGA:
    def __init__(
        self,
        objective: ObjectiveFn,
        bounds: np.ndarray,
        ineqs: Optional[IneqFn] = None,
        eqs: Optional[EqFn] = None,
        config: GAConfig = GAConfig(),
    ):
        self.f = objective
        self.ineqs = ineqs
        self.eqs = eqs
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.bounds = np.array(bounds, dtype=float)
        assert self.bounds.shape[1] == 2, "bounds must be (n_vars, 2)"
        self.n = self.bounds.shape[0]
        self.elite_count = max(1, int(np.ceil(self.cfg.elite_fraction * self.cfg.pop_size)))

    # ---------- Constraint handling (Deb's feasibility rules) ----------
    def constraint_violation(self, x: np.ndarray) -> Tuple[float, float]:
        """Returns tuple (sum_ineq_violation, sum_eq_violation)
        Inequalities g(x) <= 0 -> positive parts are violations.
        Equalities h(x)=0 -> absolute residuals, later compared to eq_tol.
        """
        g_violation = 0.0
        h_violation = 0.0
        if self.ineqs is not None:
            g_vals = np.asarray(list(self.ineqs(x)), dtype=float)
            g_violation = float(np.sum(np.maximum(g_vals, 0.0)))
        if self.eqs is not None:
            h_vals = np.asarray(list(self.eqs(x)), dtype=float)
            # Only residual beyond tolerance counts as violation
            h_violation = float(np.sum(np.maximum(np.abs(h_vals) - self.cfg.eq_tol, 0.0)))
        return g_violation, h_violation

    def is_feasible(self, x: np.ndarray) -> bool:
        g, h = self.constraint_violation(x)
        return (g <= 0.0 + 1e-16) and (h <= 0.0 + 1e-16)

    def dominates(self, a: Tuple[np.ndarray, float, float, float], b: Tuple[np.ndarray, float, float, float]) -> bool:
        """Return True if 'a' is preferred to 'b' by Deb's rules.
        Each tuple is (x, f(x), g_violation, h_violation).
        Rules:
          1) If one is feasible and the other not -> feasible wins.
          2) If both feasible -> lower objective wins.
          3) If both infeasible -> lower total violation wins.
        """
        feas_a = (a[2] + a[3]) == 0.0
        feas_b = (b[2] + b[3]) == 0.0
        if feas_a and not feas_b:
            return True
        if feas_b and not feas_a:
            return False
        if feas_a and feas_b:
            return a[1] < b[1]
        # both infeasible: compare total violation
        return (a[2] + a[3]) < (b[2] + b[3])

    # ---------- Variation operators ----------
    def _sample_uniform(self, size: Tuple[int, int]) -> np.ndarray:
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return self.rng.uniform(low, high, size=size)

    def tournament_select(self, population: List[Tuple[np.ndarray, float, float, float]]) -> np.ndarray:
        k = self.cfg.tournament_k
        idxs = self.rng.choice(len(population), size=k, replace=False)
        best = population[idxs[0]]
        for i in idxs[1:]:
            cand = population[i]
            if self.dominates(cand, best):
                best = cand
        return best[0].copy()

    def sbx_crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.cfg.crossover_prob:
            return p1.copy(), p2.copy()
        n_c = self.cfg.sbx_eta
        u = self.rng.random(self.n)
        beta = np.where(u <= 0.5, (2*u)**(1/(n_c+1)), (1/(2*(1-u)))**(1/(n_c+1)))
        child1 = 0.5*((1+beta)*p1 + (1-beta)*p2)
        child2 = 0.5*((1-beta)*p1 + (1+beta)*p2)
        return self._clip(child1), self._clip(child2)

    def polynomial_mutation(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        for i in range(self.n):
            if self.rng.random() < self.cfg.mutation_prob:
                yl, yu = self.bounds[i]
                if yl == yu:
                    continue
                delta1 = (y[i] - yl) / (yu - yl)
                delta2 = (yu - y[i]) / (yu - yl)
                rnd = self.rng.random()
                mut_pow = 1.0 / (self.cfg.mut_eta + 1.0)
                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0*rnd + (1.0 - 2.0*rnd) * (xy**(self.cfg.mut_eta + 1))
                    deltaq = (val**mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0*(1.0 - rnd) + 2.0*(rnd - 0.5) * (xy**(self.cfg.mut_eta + 1))
                    deltaq = 1.0 - (val**mut_pow)
                y[i] = y[i] + deltaq * (yu - yl)
        return self._clip(y)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, self.bounds[:,0]), self.bounds[:,1])

    # ---------- Core loop ----------
    def evaluate(self, X: np.ndarray) -> List[Tuple[np.ndarray, float, float, float]]:
        out: List[Tuple[np.ndarray, float, float, float]] = []
        for x in X:
            fx = float(self.f(x))
            gv, hv = self.constraint_violation(x)
            out.append((x.copy(), fx, gv, hv))
        return out

    def run(self, verbose: bool = True) -> Tuple[np.ndarray, float, dict]:
        # Initialize
        X = self._sample_uniform((self.cfg.pop_size, self.n))
        population = self.evaluate(X)

        def best_of(pop):
            b = pop[0]
            for cand in pop[1:]:
                if self.dominates(cand, b):
                    b = cand
            return b

        history = {"best_f": [], "best_is_feasible": [], "best_violation": []}

        for gen in range(self.cfg.generations):
            # Sort by dominance for elitism
            population.sort(key=lambda ind: (ind[2] + ind[3] > 0, ind[1] if ind[2]+ind[3]==0 else ind[2]+ind[3]))
            elites = [p[0].copy() for p in population[:self.elite_count]]

            # Create offspring
            offspring = []
            while len(offspring) < self.cfg.pop_size - self.elite_count:
                p1 = self.tournament_select(population)
                p2 = self.tournament_select(population)
                c1, c2 = self.sbx_crossover(p1, p2)
                c1 = self.polynomial_mutation(c1)
                c2 = self.polynomial_mutation(c2)
                offspring.append(c1)
                if len(offspring) < self.cfg.pop_size - self.elite_count:
                    offspring.append(c2)

            # Next population = elites + offspring
            X_next = np.vstack([np.array(elites), np.array(offspring)])
            population = self.evaluate(X_next)

            b = best_of(population)
            history["best_f"].append(b[1])
            history["best_is_feasible"].append((b[2] + b[3]) == 0.0)
            history["best_violation"].append(b[2] + b[3])

            if verbose and (gen % max(1, self.cfg.generations // 10) == 0 or gen == self.cfg.generations - 1):
                status = "feasible" if (b[2]+b[3]) == 0.0 else f"infeasible (viol={b[2]+b[3]:.3e})"
                print(f"Gen {gen:4d}: best f = {b[1]:.6f} [{status}]")

        best = min(population, key=lambda ind: (ind[2] + ind[3] > 0, ind[1] if ind[2]+ind[3]==0 else ind[2]+ind[3]))
        x_best, f_best, g_best, h_best = best
        info = {
            "feasible": (g_best + h_best) == 0.0,
            "violation": g_best + h_best,
            "history": history,
            "config": self.cfg,
        }
        return x_best, f_best, info

# ---------------------- Example problem ----------------------
# Non-convex Rastrigin objective with nonlinear constraints
#   minimize f(x) = sum_i [10 + x_i^2 - 10 cos(2π x_i)]
#   s.t. g1(x) = x0^2 + x1^2 - 5 <= 0   (inside circle radius sqrt(5))
#        g2(x) = 1 - (x0 + x1) <= 0     (x0 + x1 >= 1)
#        h1(x) = x0 - x1 = 0            (approximately equal with tolerance)

def rastrigin(x: np.ndarray) -> float:
    return float(np.sum(10 + x**2 - 10*np.cos(2*np.pi*x)))

def ineqs_example(x: np.ndarray):
    x0, x1 = x[0], x[1]
    return [x0**2 + x1**2 - 5.0, 1.0 - (x0 + x1)]

def eqs_example(x: np.ndarray):
    return [x[0] - x[1]]

if __name__ == "__main__":
    dim = 2
    bounds = np.array([[-3.5, 3.5], [-3.5, 3.5]])

    cfg = GAConfig(
        pop_size=150,
        generations=200,
        tournament_k=3,
        crossover_prob=0.9,
        mutation_prob=0.1,
        sbx_eta=15,
        mut_eta=20,
        elite_fraction=0.03,
        eq_tol=1e-3,
        seed=7,
    )

    ga = ConstrainedGA(
        objective=rastrigin,
        bounds=bounds,
        ineqs=ineqs_example,
        eqs=eqs_example,
        config=cfg,
    )

    x_best, f_best, info = ga.run(verbose=True)
    print("\nBest solution:")
    print("x* =", x_best)
    print("f(x*) =", f_best)
    print("feasible =", info["feasible"], ", total violation =", info["violation"])
