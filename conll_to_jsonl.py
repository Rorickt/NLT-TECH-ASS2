
from collections import defaultdict
import csv
import json


def preprocessed_conll_to_dict(preprocessed_conllfile):

    """
    Extracts the tokens, arguments, and predicate information per sentence

    Args:
        A conllfile that has been preprocessed to the extent where each sentence
        is duplicated for every predicate it contains.

    """

    all_sentences = []
    sentence_tokens = []

    sentence_arguments = []
    all_sentence_args = []

    sentence_predicates = []
    all_sentence_preds = []

    token_dict_list = []


    with open(preprocessed_conllfile, "r", encoding = "utf8") as file:
        conllfile = csv.reader(file, delimiter='\t', quotechar='^')

        for row in conllfile:
            if row == []:
                all_sentences.append(sentence_tokens)
                sentence_tokens = []
                all_sentence_args.append(sentence_arguments)
                sentence_arguments = []
                all_sentence_preds.append(sentence_predicates)
                sentence_predicates = []

            else:
                if row[0].startswith('#') or row[0].startswith('"#'):
                    continue
                else:
                    if row[1].startswith("[") or row[1].startswith("]"):
                        continue
                    else:
                        sentence_tokens.append(row[1])
                        if row[-1].startswith("ARG"):
                            sentence_arguments.append("B-"+row[-1])
                        elif row[-1] == "V":
                            sentence_arguments.append("B-"+row[-1])
                        else:
                            sentence_arguments.append("O")
                        if row[10] != "_":
                            sentence_predicates.append(int(row[6]))
                            sentence_predicates.append(row[10])
                            sentence_predicates.append("V")
                            sentence_predicates.append(row[4])

    for item in zip(all_sentences, all_sentence_args, all_sentence_preds):
        token_dict = {"seq_words":item[0], "BIO": item[1],
                     "pred_sense":item[2]}
        token_dict_list.append(token_dict)

    return token_dict_list


def write_to_jsonfile(list_of_dicts, file_path):

    """
    Take a list of dictionaries that contains information on tokens, arguments,
    and predicates per sentence, and writes it to a json file
    Args:
        A python object, here a list of dicts is needed.
    """

    fp = file_path.replace('.conllu', '.jsonl')
    jsonfile = json.dumps(list_of_dicts)

    with open(fp, 'w') as outfile:

        for entry in list_of_dicts:
            json.dump(entry, outfile)
            outfile.write('\n')




def run_preprocessing(file_path):
    """
    Run functions for extracting tokens, args and predicate information and
    writing the outcome to a json file

    Args:
        preprocessed file: a preprocessed conll-file
    """

    list_of_dicts = preprocessed_conll_to_dict(file_path)

    write_to_jsonfile(list_of_dicts, file_path)

run_preprocessing('en_ewt-up-train_preprocessed.conllu') #converting preprocessed train data
run_preprocessing('en_ewt-up-dev_preprocessed.conllu') #converting preprocessed dev data
