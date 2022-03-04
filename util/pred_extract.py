# All functions for Predicate extraction
import csv

def extract_predicates(data, data_type):
    """ returns, and stores, all extracted predicates based on POS tag of the token
    
    Args:
        data: filepath to the preprocessed dev/test file
        data_type: string of 'dev' or 'test'

    Returns:
        pred_list_eval: a list of all labels per token (for evaluation) 
    """
    pred_list_eval=[]
    with open(data, 'r', encoding='utf8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='^')
        
        # ready output file
        output_filepath = 'output/'+data_type+'_predicate_extraction.conllu'
        with open(output_filepath, 'w', newline='', encoding='utf8') as csvfile:
            output = csv.writer(csvfile, delimiter='\t')

            for token in file:
                if token == []:
                    output.writerow(token)
                else:
                    if token[0].startswith('#') or token[0].startswith('"#'):
                        output.writerow(token)

                    # write token rows with only one predicate and the correlating column of arguments
                    else:
                        new_row = token[:10] # all columns up till sense info, excluding sense info
                        if token[3] == 'VERB':
                            new_row.append(token[2]) # lemma
                            pred_list_eval.append('V')
                        else:
                            new_row.append('_') # underscore for every other token that is not predicate
                            pred_list_eval.append('_')
                        output.writerow(new_row)
    return pred_list_eval

def gold_predicate_labels(data):
    """ returns, and stores, all predicates from the gold data
    
    Args:
        data: filepath to the preprocessed dev/test file

    Returns:
        gold_predlist_eval: a list of all gold labels per token (for evaluation) 
    """
    gold_predlist_eval=[]
    with open(data, 'r', encoding='utf8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='^')
        for token in file:
            if token == []:
                continue
            else:
                if token[0].startswith('#') or token[0].startswith('"#'):
                    continue
                else:
                    if token[10] != '_':
                        gold_predlist_eval.append('V')
                    else:
                        gold_predlist_eval.append('_')
        
    return gold_predlist_eval