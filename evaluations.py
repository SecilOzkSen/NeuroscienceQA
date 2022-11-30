import jury
from jury import load_metric, Jury

SQUAD = [load_metric("squad", resulting_name="squad")]

mt_scorer = Jury(metrics=SQUAD, run_concurrent=True)
#scores = mt_scorer(predictions=mt_predictions, references=mt_references)

if __name__ == '__main__':
    print(jury.list_metrics())