import json
import os
import pandas as pd
import numpy as np
results_dir = '.'

if __name__ == "__main__":
    # list all the jsonl files in the results directory
    jsonl_files = [f for f in os.listdir(results_dir) if f.endswith('.jsonl')]

    all_data = []
    for file in jsonl_files:
        # peek at the first line to get the keys
        agg_data = dict()
        with open(os.path.join(results_dir, file), 'r') as f:
            sample = json.loads(f.readline())
            for k,v in sample.items():
                if type(v) == float or type(v) == int:
                    agg_data[k] = []
                if type(v) == bool:
                    agg_data[k] = []
            
        with open(os.path.join(results_dir, file), 'r') as f:
            for line in f:
                data = json.loads(line)
                for k in agg_data.keys():
                    agg_data[k].append(data[k])
        
        # now summarize what we have
        for k in agg_data.keys():
            agg_data[k] = np.mean(agg_data[k])
        agg_data['file'] = file
        all_data.append(agg_data)

    # now save the data to a csv
    df = pd.DataFrame(all_data)
    df.to_csv('morehopqa_results_summary.csv', index=False)

    print(f"Saved results to morehopqa_results_summary.csv")
