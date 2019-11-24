import json
import sys
import re
import random
import copy
from collections import defaultdict


def reformat_equation(equation):
    def process_percent(s):
        matches = re.finditer(r'(\d+\.?\d*)%', s)
        for match in matches:
            percent = round(float(match.group(1)) * 0.01, 3)
            s = s.replace(match.group(), str(percent), 1)

        return s

    equation = process_percent(equation)
    if 'is_' in equation or 'be_' in equation:  # univariate logic expression
        return equation
    equation = equation.replace(',', '')  # remove commas
    equation = equation.replace('\u221a', 'sqrt')
    equation = equation.replace('\u03c0', 'pi')
    # equation = equation.replace('\u00b0', '*pi/180')  # change degree to rad

    # change them to upper case just to avoid inserting '*'
    equation = equation.replace('sin', 'SIN')
    equation = equation.replace('cos', 'COS')
    equation = equation.replace('tan', 'TAN')
    equation = equation.replace('cot', 'COT')
    equation = equation.replace('sqrt', 'SQRT')
    equation = equation.replace('pi', 'PI')

    # add multiplication operator
    positions = []
    for i, char in enumerate(equation):
        if i > 0 and re.match(r'[a-zA-Z\(]', char) and re.match(r'[0-9a-z\)]', equation[i-1]):
            positions.append(i)
    for i, pos in enumerate(positions):
        equation = equation[:pos+i] + '*' + equation[pos+i:]

    equation = replace_variables(equation)

    return var_on_left(equation.lower())


def var_on_left(equation):
    sides = equation.split("=")
    if len(sides) == 1:
        return equation
    if re.search(r'[a-z]', sides[0]) and len(sides[0]) < len(sides[1]):
        return equation
    if re.search(r'[a-z]', sides[1]):
        return '{}={}'.format(sides[1], sides[0])
    return equation


def replace_variables(equation):
    variables = set(re.findall(r'[a-z]', equation))
    if 'x' in variables:
        return equation

    n_var = len(variables)
    if n_var > 3:
        return equation

    xyz = ('x', 'y', 'z')
    vars = sorted(list(variables))
    for i in range(n_var):
        equation = equation.replace(vars[i], xyz[i])

    return equation

if __name__ == '__main__':
    jsonfile = sys.argv[1]
    destination = sys.argv[2]
    data = json.load(open(jsonfile))
    new_data = []

    for item in data:
        d = {}
        d['ans'] = item['ans']
        d['text'] = item['text'].split('\"')[0]
        # d['text'] = item['original_text']
        d['id'] = item['id']

        # parse equations
        equations = item['equations'].split('\r\n')
        equs = []
        for part in equations:
            if part[:5] == 'unkn:':
                d['unkn'] = part.replace('unkn:', '').replace(' ', '')
            elif part[:4] == 'equ:':
                part = part.replace('equ:', '').replace(' ', '')
                equs.append(reformat_equation(part))
            elif part:
                equs.append(reformat_equation(part))
                print("equation without new line", d['id'], part)
        d['equations'] = equs
        new_data.append(d)

        random.shuffle(new_data)

    with open(destination, 'w') as f:
        json.dump(new_data, f, indent=2)



