import json
import sys
import re
import copy


def reformat_equation(equation):
    if 'is_' in equation or 'be_' in equation:  # univariate logic expression
        return equation
    equation = equation.replace(',', '')  # remove commas
    equation = equation.replace('%', '*.01')
    equation = equation.replace('sin', 'SIN')
    equation = equation.replace('cos', 'COS')
    equation = equation.replace('tan', 'TAN')
    equation = equation.replace('cot', 'COT')

    # add multiplication operator
    positions = []
    for i, char in enumerate(equation):
        if i > 0 and re.match(r'[a-zA-Z\(]', char) and re.match(r'[0-9a-z\)]', equation[i-1]):
            positions.append(i)
    for i, pos in enumerate(positions):
        equation = equation[:pos+i] + '*' + equation[pos+i:]

    equation = equation.replace('\u221a', 'sqrt')
    equation = equation.replace('\u03c0', 'pi')
    equation = equation.replace('\u00b0', '*pi/180')  # change degree to rad
    return equation.lower()

if __name__ == '__main__':
    jsonfile = sys.argv[1]
    destination = sys.argv[2]
    data = json.load(open(jsonfile))
    new_data = []

    for item in data:
        d = {}
        d['ans'] = item['ans']
        d['text'] = item['text']
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

    with open(destination, 'w') as f:
        json.dump(new_data, f, indent=2)



