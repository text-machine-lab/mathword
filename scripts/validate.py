import sys
sys.path.append('..')

import json
import sys
import traceback
from src.check_answers import check_solution
from scripts.preprocess import reformat_equation

data = json.load(open(sys.argv[1]))
errors = 0
cannot_check = 0
for i, d in enumerate(data):
    answer = d['ans']
    equations = d['equations']
    if not equations or not answer:
        continue
    unkns = d['unkn']
    unkns = unkns.replace(';', ',')
    unkns = unkns.split(',')

    # some fixes
    if len(equations) == 1 and len(unkns) == 2:
        new_equation = reformat_equation('y={}'.format(unkns[1]))
        equations.append(new_equation)
    if len(unkns) == 1:
        answer = answer.replace(';', 'or')

    # try:
    score, solution = check_solution(answer, equations)
    # except SyntaxError:
    #     print(d)
    #     traceback.print_exc()
    #     sys.exit(1)
    if score == 0 and solution is None:
        cannot_check += 1
    elif score != 1.0:
        errors += 1
        print(score, answer, solution, equations, d['id'])
    sys.stdout.write('%d questions processed\r' % i)

print("\n{} out of {} errors".format(errors, len(data)))
print("{} out of {} cannot check".format(cannot_check, len(data)))
