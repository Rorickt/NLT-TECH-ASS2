import csv, pandas as pd
from sklearn.metrics import classification_report

#Gold
gold_labels=[]
with open('data/en_ewt-up-dev_preprocessed.conllu', encoding='utf-8') as file:
    file = csv.reader(file, delimiter='\t', quotechar='"')
    for row in file:
        if row == []:
            continue
        
        # if token is a predicate: append to list of predicates of current sentence
        elif row[0].startswith('#') or row[0].startswith('"'):
            continue
                
        else: 
            if row[-1] == 'V':
                gold_labels.append('V')
            elif row[-1] == '_':
                gold_labels.append('_')
            else: 
                gold_labels.append('ARG')


# Extracted labels
machine_labels=[]
with open('output/dev_extract_arguments.conllu', encoding='utf-8') as file:
    file = csv.reader(file, delimiter='\t', quotechar='"')
    for row in file:
        if row == []:
            continue
        
        # if token is a predicate: append to list of predicates of current sentence
        elif row[0].startswith('#') or row[0].startswith('"'):
            continue
                
        else: 
            if row[-1] == 'V':
                machine_labels.append('V')
            elif row[-1] == '_':
                machine_labels.append('_')
            else: 
                machine_labels.append('ARG')


print(len(machine_labels))

## Step 3: Evaluate extraction
argument_eval = classification_report(y_true=gold_labels,
                                        y_pred=machine_labels,
                                        output_dict=True)

arg_eval_df = pd.DataFrame(argument_eval).T
arg_eval_df.to_latex('output/arg_extract_eval.txt')
print(arg_eval_df)