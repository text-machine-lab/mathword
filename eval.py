import json
import sys
from tqdm import tqdm
from src.check_answers import check_solution


def main():
    pred_file = sys.argv[1]
    with open(pred_file) as f:
        pred = json.load(f)

    n = len(pred)
    all_scores = 0
    for d in tqdm(pred, mininterval=2, leave=False):
        answer = d['ans']
        # equations = d['equation'].split(';')
        equations = d['pred'][0].split(';')
        if 'pred_2' in d and d['pred_2'][1] > d['pred'][1]:
            equations = d['pred_2'][0].split(';')

        try:
            score, solution = check_solution(answer, equations)
        except:
            score = 0
        all_scores += round(score + 0.1)
    # print("Truth accuracy: {:.3f}  -- {} out of {} correct.".format(all_scores/n, int(all_scores), n))
    print("Solution accuracy: {:.3f}  -- {} out of {} correct.".format(all_scores / n, int(all_scores), n))

if __name__ == '__main__':
    main()
