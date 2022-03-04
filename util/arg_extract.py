# all functions for Argument extraction
import csv
from collections import Counter


def allowed_dep_paths(featuredata):
    """ Returns a list of common argument dependency paths
    Args:
        featuredata: a conllu of the feature extractions including gold labels
        data_type: a string of 'train' / 'dev' / 'test' representing the data type
    
    Returns:
        final_allowed_paths: a list of strings of the allowed dependency paths
    """
    
    allowed_paths=list()
    disallowed_paths=list()
    with open(featuredata, 'r', encoding='utf-8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='"')
        
        sentence_num = 0
        for token in file:
            if token == []:
                sentence_num += 1
                continue

            elif token[0].startswith('#') or token[0].startswith('"'):
                    continue

            else:
                dependencypath = token[9]
                if token[-1] == 'V':
                    disallowed_paths.append(dependencypath)
                elif token[-1] == '_':
                    disallowed_paths.append(dependencypath)
                else:
                    allowed_paths.append(dependencypath)

    final_allowed_paths = list()
    
    ordered_paths = Counter(allowed_paths).most_common(20)
    for path, freq in ordered_paths:
        final_allowed_paths.append(path)

    return final_allowed_paths


def write_arg_prediction(preprocessed_data, features, set_allowed_paths, data_type):
    """ Returns a list of argument labels extracted via dependency rules
    
    Args:
        preprocessed_data: a conllu of the preprocessed gold data
        features: a conllu of the extracted features of the same date type
        allowed_paths: a list of strings representing common argument dependency paths
        data_type: a string representing the current data type ('train'/'test'/'dev')
    
    """
    with open(preprocessed_data, 'r', encoding='utf-8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='"')

        with open(features, 'r', encoding='utf-8') as dep_file:
            dependency_paths = csv.reader(dep_file, delimiter='\t', quotechar='"')

            output_filepath = 'output/'+data_type+'_extract_arguments.conllu'
            with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                output = csv.writer(csvfile, delimiter='\t')

                for gold, dep in zip(file, dependency_paths):
                    
                    if gold == []:
                        output.writerow(gold)

                    elif gold[0].startswith('#') or gold[0].startswith('"'):
                        output.writerow(gold) 
                    
                    else:
                        new_row = gold[:11]
                        # if current token is predicate, it cannot be arg
                        if dep[-2]!= '_':
                            new_row.append('V')
                        else:
                            if dep[-3] in set_allowed_paths:
                                new_row.append('ARG')
                            else:
                                new_row.append('_')
                                
                        output.writerow(new_row)


def argument_extraction(gold_data, pred_label_data):
    """Get the gold and machine labels from the output files
    
    Args:
        gold_data: a conllu of the preprocessed gold data
        pred_label_data: a conllu of the predicted labels of the same date type
    """

    gold_labels=[]
    with open(gold_data, encoding='utf-8') as file:
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
    with open(pred_label_data, encoding='utf-8') as file:
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

    return machine_labels, gold_labels