import numpy as np
import os
import os.path as op
import pandas as pd

from rouge import Rouge


rouge_scorer = Rouge()


def get_metrics_dictionary(exp_name, epoch_name):
    
    out_dict = {}
    
    out_dict["experiment"] = exp_name
    out_dict["epoch"] = int(epoch_name.split("_")[0])
    
    results_df = pd.read_csv(op.join("experiments", exp_name, epoch_name))

    references = list(results_df["gold_summary"])
    references = [[reference] for reference in references]

    predictions = list(results_df["predicted_summary"])

    scores = rouge_scorer._compute(predictions, references)

    for metric_name, score in scores.items():
        out_dict[metric_name] = score

    return out_dict




if __name__ == '__main__':
    


    scores_list = []
    experiments = os.listdir("experiments")
    for i, exp_name in enumerate(experiments):
        print(f"Scoring experiment {i+1} of {len(experiments)}")
        exp_dir = op.join("experiments", exp_name)
        scores_files = [f for f in os.listdir(exp_dir) if f.endswith("epochs.csv")]
        for j, epoch_name in enumerate(scores_files):
            scores_list.append(get_metrics_dictionary(exp_name, epoch_name))
            print(f"File {j+1} of {len(scores_files)} done in {exp_name}")            


    scores_df = pd.DataFrame(scores_list)
    for metric in metrics_dict:
        scores_df[metric] = pd.to_numeric(scores_df[metric], errors="coerce")
    scores_df.to_csv("all_scores.csv")
    print(scores_df.sort_values("Rouge2", ascending=False))
    
