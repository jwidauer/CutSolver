#!/usr/bin/env python3

from typing import Dict, List, Tuple

import numpy as np
from pulp import (
    LpProblem,
    LpConstraintVar,
    LpMinimize,
    LpConstraintGE,
    LpVariable,
    LpContinuous,
    LpInteger,
    lpSum,
    PULP_CBC_CMD,
    value,
)


class StockCutSolver:
    """
    Optimal solver for the stock cutting problem.
    Optimizes for lowest cost.
    """

    # Problem parameters
    demand: Dict[float, int]
    cut_width: float
    max_length: float
    cost: float

    problem: LpProblem

    patterns: Dict[str, Dict[float, int]] = {}

    def __init__(
        self,
        demand: Dict[float, int],
        cut_width: float,
        max_length: float,
        stock_cost: float,
    ):
        """
        docstring
        """
        self.demand = demand
        self.cut_width = cut_width
        self.max_length = max_length
        self.cost = stock_cost

        self.problem = LpProblem("MasterStockCuttingProblem", LpMinimize)

        self.objective = LpConstraintVar("Obj")
        self.problem.setObjective(self.objective)

        self.constraints = dict()
        for length in demand:
            self.constraints[length] = LpConstraintVar(
                "C" + str(length), LpConstraintGE, demand[length]
            )
            self.problem += self.constraints[length]

    def solve(self) -> Tuple[float, List[Tuple[int, Dict[float, int]]]]:
        """
        Optimally solves the stock cutting problem.
        """

        # A list of starting patterns is created
        new_patterns = np.eye(len(self.demand), dtype=int).tolist()

        # New patterns will be added until new_patterns is an empty list
        while new_patterns:
            # The new patterns are added to the problem
            self.addPatterns(new_patterns)
            # The master problem is solved, and the dual variables are returned
            duals = self.masterSolve()
            # The sub problem is solved and a new pattern will be returned if there is one
            # which can reduce the master objective function
            new_patterns = self.subSolve(duals)

        # The master problem is solved with Integer constraints not relaxed
        self.masterSolve(relax=False)

        solution = [
            (v.varValue, self.patterns[v.name])
            for v in self.problem.variables()
            if v.varValue != 0
        ]

        return value(self.problem.objective), solution

    def addPatterns(self, new_patterns):
        """
        Adds new patterns to the master problem.
        """

        for pattern in new_patterns:
            # The new patterns are checked to see that their length does not exceed
            # the total roll length
            lsum = (
                np.dot(pattern, list(self.demand)) + (sum(pattern) - 1) * self.cut_width
            )

            if lsum > self.max_length:
                raise ("Length option too large!")

            # Create the pattern dictionary and name
            pattern_name = "P" + str(len(self.patterns))
            pattern_dict = dict(zip(list(self.demand), pattern))
            self.patterns[pattern_name] = pattern_dict

            # Create the problem variable and add it to the problem
            LpVariable(
                pattern_name,
                0,
                None,
                LpContinuous,
                self.cost * self.objective
                + lpSum([self.constraints[l] * pattern_dict[l] for l in self.demand]),
            )

    def masterSolve(self, relax: bool = True):
        """
        Solve the master problem.
        If `relax` is False, the variables will be assumed as integer.
        """
        # Unrelaxes the Integer Constraint
        if not relax:
            for variable in self.problem.variables():
                variable.cat = LpInteger

        # The problem is solved and rounded
        self.problem.solve(PULP_CBC_CMD(msg=0))
        self.problem.roundSolution()

        # A dictionary of dual variable values is returned
        return {l: self.problem.constraints["C" + str(l)].pi for l in self.demand}

    def subSolve(self, duals) -> List[List[int]]:
        """
        Solve the sub-problem of finding a pattern that would reduce the master
        problems objective.
        """
        # The sub problem is created
        prob = LpProblem("SubProb", LpMinimize)

        # The problem variables are created
        vars = LpVariable.dicts("length", list(self.demand), 0, None, LpInteger)

        trim = LpVariable("trim", 0, None, LpContinuous)

        # The objective function is entered: the reduced cost of a new pattern
        prob += (
            self.cost - lpSum([vars[i] * duals[i] for i in self.demand]),
            "Objective",
        )

        # The conservation of length constraint is entered
        prob += (
            lpSum([vars[l] * l for l in self.demand])
            + (lpSum([vars[i] for i in self.demand]) - 1) * self.cut_width
            + trim
            == self.max_length,
            "lengthEquate",
        )

        # The problem is solved
        prob.solve(PULP_CBC_CMD(msg=0))

        # The variable values are rounded
        prob.roundSolution()

        new_patterns = []
        # Check if there are more patterns which would reduce the master LP
        # objective function further
        if value(prob.objective) < -(10 ** -5):
            varsdict = {v.name: v.varValue for v in prob.variables()}

            # Adds the new pattern to the newPatterns list
            pattern = [int(varsdict["length_" + str(l)]) for l in self.demand]
            new_patterns.append(pattern)

        return new_patterns


if __name__ == "__main__":
    demand = {5: 150, 7: 200, 9: 300}

    solver = StockCutSolver(demand, 0.5, 20, 1)
    total_cost, patterns = solver.solve()

    for pattern in patterns:
        print([pattern[1][l] for l in demand], "x", pattern[0])

    print("Total cost:", total_cost)
