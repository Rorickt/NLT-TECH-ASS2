# all functions for Argument extraction
import csv
from collections import Counter


def allowed_paths_gold(featuredata, data_type):
    """ Returns a list of common argument dependency paths,
        and gold argument labels
    Args:
        featuredata: a conllu of the feature extractions including gold labels
        data_type: a string of 'train' / 'dev' / 'test' representing the data type
    
    Returns:
        final_allowed_paths: a list of strings of the allowed dependency paths
        gold_args: a list of gold argument labels ('_','V','ARG')
    """
    
    gold_args = []
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
                    gold_args.append('V')

                elif token[-1] == '_':
                    disallowed_paths.append(dependencypath)
                    gold_args.append('_')
                
                else:
                    allowed_paths.append(dependencypath)
                    gold_args.append('ARG')


    final_allowed_paths = list()
    
    ordered_paths = Counter(allowed_paths).most_common(20)
    for path, freq in ordered_paths:
        final_allowed_paths.append(path)

    
    return final_allowed_paths, gold_args


def extract_arguments(preprocessed_data, features, set_allowed_paths, data_type):
    """ Returns a list of argument labels extracted via dependency rules
    
    Args:
        preprocessed_data: a conllu of the preprocessed gold data
        features: a conllu of the extracted features of the same date type
        allowed_paths: a list of strings representing common argument dependency paths
        data_type: a string representing the current data type ('train'/'test'/'dev')
    
    Returns:
        extracted_args: a list of extracted argument labels ('_','V','ARG')
    """
    
    extracted_args=[]
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
                            extracted_args.append('V')
                        else:
                            if dep[-3] in set_allowed_paths:
                                new_row.append('ARG')
                                extracted_args.append('ARG')
                            else:
                                new_row.append('_')
                                extracted_args.append('_')
                        output.writerow(new_row)


    return extracted_args

