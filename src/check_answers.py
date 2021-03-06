import re
import logging
from sympy import Eq, symbols, sin, cos, tan, cot, pi, E, factorial, lcm, gcd, sqrt
from sympy.solvers import solve
from sympy.parsing.sympy_parser import parse_expr
from fractions import Fraction
import numpy as np
import traceback
import sys

DECIMALS = 3

def replace_perm(equation):
    """p(n,m) --> factorial(n)/factorial(m)"""

    matches = re.findall(r'([Pp]\(([\d\+\-]+),([\d]\+\-+)\))', equation)
    if not matches:
        return equation

    for match in matches:
        P_str = match[0]
        n = match[1]
        m = match[2]
        equation = equation.replace(P_str, '(factorial({})/factorial({}))'.format(n, m))
    return equation


def replace_comb(equation):
    """c(n,m) --> factorial(n)/(factorial(m)*factorial(n-m))"""

    matches = re.findall(r'([Cc]\(([\d\+\-]+),([\d\+\-]+)\))', equation)
    if not matches:
        return equation

    for match in matches:
        C_str = match[0]
        n = match[1]
        m = match[2]
        equation = equation.replace(C_str, '(factorial({})/(factorial({})*factorial({}-{})))'.format(n, m, n, m))
    return equation

def solve_equations(equations):
    """
    :param equations: list of strings e.g. ['x + y = 48', 'x = 3*y']
    :return: solveset
    """
    if type(equations) == str:
        equations = [equations]
    eqs = []
    variables = set([])
    for equation in equations:

        # remove inequalities, not able to process them for now
        if '<' in equation or '>' in equation:
            continue
        if 'is_' in equation or 'be_' in equation: # constraints
            continue
        equation = equation.replace('e', 'E') # sympy uses E for natural log base
        equation = replace_comb(replace_perm(equation))

        if equation.count('=') != 1:
            return []  # no solution
        equation = equation.replace('^', '**')
        equation = equation.replace('%', '*.01')
        equation = equation.replace('[', '(')
        equation = equation.replace(']', ')')

        left, right = equation.split('=')
        if not left or not right:
            return []
        eq = Eq(parse_expr(left), parse_expr(right))
        variables = variables.union(eq.free_symbols)
        eqs.append(eq)

    if len(variables) > len(equations):
        return []
    try:
        if len(eqs) == 1:  # have to do this for certain functions, may be a sympy bug
            solutions = solve(eqs[0], variables)
        else:
            solutions = solve(eqs, variables)
        return solutions
    except:
        return []


def parse_answer(answer, decimals=DECIMALS):
    answer = answer.strip()
    if '|' in answer:
        choices = answer.split('|')
    else:
        choices = [answer]

    ansformats = []  # equivalent answer formats
    for choice in choices:
        choice = choice.strip()
        if 'or' in choice:
            valid_answers = choice.split('or')
        else:
            valid_answers = [choice]
        ansset = []  # distinctive answers to a question
        for ans in valid_answers:
            ans = ans.strip()
            ans = re.sub('[a-zA-Z$]+[\^2-3]*', '', ans)
            ans = re.sub('\s+/', '', ans)

            # for now, answers with {} or not are not distinguished. may handle it in the future
            if '{' in ans:
                vals = ans.strip('{}').split(';')
            else:
                vals = ans.split(';')
            if '' in vals:
                vals.remove('')  # remove empty elements
            try:
                vals = sorted([round(eval(x), decimals) for x in vals])
            except:
                # logging.warning("cannot convert to real values: {}".format(vals))
                # print(vals)
                continue
            ansset.append(vals)
        ansformats.append(ansset)

    # [list of equivalent answers [sets of distinct values [values ..] ] ]
    return ansformats[0]  # for now just return one answer


def check_solution(answer, equations, decimals=DECIMALS, error=0.01):
    """
    solve equations and check if the solution is the same as the answer
    :param answer: string e.g. "5/3 or -3/5 | 1.667 or -0.6"
    :param equations: list of strings e.g. ["n - 1/n = 16/15"]
    :return: bool

    note: items like sqrt(5) is in type sympy.core.power.Pow
    and will be converted to float with float(sqrt(5))
    """
    try:
        solutions = solve_equations(equations)
    except:
        # print(answer, equations)
        # traceback.print_exc()
        # sys.exit(1)
        return 0, []
    if solutions == []:
        return 0, []
    ans = parse_answer(answer)
    if type(solutions) == list:  # multiple distinct solutions
        n_solutions = len(solutions)
        try:
            assert len(ans) <= n_solutions  # <= because valid answers can be a subset of equation roots
        except AssertionError:
            # print(answer, equations)
            # traceback.print_exc()
            # sys.exit(1)
            return 0, None
    else:
        solutions = [solutions]
        n_solutions = 1

    points = 0
    for solution in solutions:  # for each disctinct solution
        try:
            if type(solution) == dict:
                solution_vals = sorted([round(float(v), decimals) for k, v in solution.items()])
            else:
                solution_vals = [round(float(solution), decimals)]
        except:
            # print(answer, equations, solution)
            # traceback.print_exc()
            return 0, None
        n_vars = len(solution_vals)
        for item in ans:  # for each valid answer
            if len(item) == n_vars:
                if np.linalg.norm(np.array(item)-np.array(solution_vals), 1) < error*n_vars:  # if error is very small
                    points += 1
                    break  # move to another solution
            elif len(item) == 1:
                for val in solution_vals:
                    if abs(val - item[0]) < error:
                        points += 1
                        break
            # else:
                # print("answer:", answer, "equations:", equations)
                # print("# answers does not match # solutions")
                # sys.exit("# answers does not match # solutions")

    return points/n_solutions, solutions


def get_score(answer, eq_str, decimals=DECIMALS, error=0.01):
    equations = eq_str.split('s')





