import json
import sys
import os
from tqdm import tqdm
import argparse
from src.check_answers import check_solution


def main():
    parser = argparse.ArgumentParser(description='predict.py')
    parser.add_argument('result_file')
    parser.add_argument('-wrong', default=None, help="output path with wrong anwers")
    parser.add_argument('-right', default=None, help="output path with right anwers")
    parser.add_argument('-truth', action='store_true', default=False, help="use truth equations for evaluation")

    args = parser.parse_args()

    pred_file = args.result_file

    with open(pred_file) as f:
        pred = json.load(f)

    n = len(pred)
    all_scores = 0
    wrong_output = []
    right_output = []
    for d in tqdm(pred, mininterval=2, leave=False):
        answer = d['ans']
        if args.truth:
            equations = d['equation'].split(';')
        else:
            equations = d['pred'][0].split(';')
            if 'pred_2' in d and d['pred_2'][1] > d['pred'][1]:
                equations = d['pred_2'][0].split(';')

        try:
            score, solution = check_solution(answer, equations)
        except:
            score = 0
        all_scores += round(score + 0.1)

        if args.wrong and score == 0:
            wrong_output.append(d)
        if args.right and score == 1:
            d["chosen_equations"] = equations
            right_output.append(d)

    print("Solution accuracy: {:.3f}  -- {} out of {} correct.".format(all_scores / n, int(all_scores), n))
    if args.wrong is not None:
        with open(os.path.join(args.wrong, 'wrong_answers.json'), 'w') as f:
            json.dump(wrong_output, f, indent=2)
        print("saved wrong answers")
    if args.right is not None:
        with open(os.path.join(args.right, 'right_answers.json'), 'w') as f:
            json.dump(right_output, f, indent=2)
        print("saved right answers")

if __name__ == '__main__':
    main()
