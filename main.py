from util import preprocess_gold, preprocess_argclass, \
                extract_features_and_labels,\
                gold_predicate_labels, extract_predicates, \
                allowed_paths_gold, extract_arguments

from sklearn.metrics import classification_report
import pandas as pd


###################################
###    Preprocess datafiles     ### (duplicate sentences per predicate)
###################################
# data_files = ['data/en_ewt-up-train.conllu', 'data/en_ewt-up-dev.conllu', 'data/en_ewt-up-test.conllu']
# for data in data_files:
#     preprocess_gold(data)
#     preprocess_argclass(data)

###################################
###     Feature extraction      ###
###################################
extract_features_and_labels('data/en_ewt-up-train_preprocessed.conllu', 'train')
extract_features_and_labels('data/en_ewt-up-dev_preprocessed.conllu', 'dev')
extract_features_and_labels('data/en_ewt-up-test_preprocessed.conllu', 'test')

###################################
###     Predicate extraction    ###
###################################
# # Step 1: with original file, find all 'VERB' in dev/test
# extracted_predicates = extract_predicates('data/en_ewt-up-dev.conllu', 'dev')
# gold_predicates = gold_predicate_labels('data/en_ewt-up-dev.conllu')

# ## Step 2: Evaluate extraction
# predicate_eval = classification_report(y_true=gold_predicates,
#                                         y_pred=extracted_predicates,
#                                         output_dict=True)

# pred_eval_df = pd.DataFrame(predicate_eval).T
# pred_eval_df.to_latex('output/pred_extract_eval.txt')

###################################
###     Argument extraction     ###
###################################
## Step 1: Prepare set of allowed dependecy paths 
allowed_paths, null = allowed_paths_gold('output/trainfeatures.conllu', data_type='train')

## Step 2: extract argument from gold and dev/test
null, gold_args = allowed_paths_gold('output/devfeatures.conllu', data_type='dev')
extracted_args = extract_arguments('data/en_ewt-up-dev_preprocessed.conllu', 
        'output/devfeatures.conllu', allowed_paths, data_type='dev')

print(len(gold_args))
print(len(extracted_args))


## Step 3: Evaluate extraction
argument_eval = classification_report(y_true=gold_args,
                                        y_pred=extracted_args,
                                        output_dict=True)

arg_eval_df = pd.DataFrame(argument_eval).T
arg_eval_df.to_latex('output/arg_extract_eval.txt')
print(arg_eval_df)


#################################
###    Classify Arguments     ###
#################################

## Step 1: extract features (use preprocessed data)
# For train data
# - for each token get: token, {features},'_'/predicate,  '_'/ARG0 (e.g.)
# For dev/test data:
# - for each token get: token, {features} '_'/predicate,  '_'/ARG  (arg is only ARG no specification)
# - store both in a 'train/dev/test_features.connlu' with same format as original (including '#...' rows)
# * we should also write out which features we use and why.

## Step 2: Train classifier
# - get list of feature_dicts, and vectorize
# - train classifier
# - run on dev/test

## Step 3: Evaluate classifier
# - get list of all arguments from gold and dev/test (should be same length)
#   probably better to skip all non-arguments, getting more informative performance scores
# - eval using sklearn?
# - store in folder: output/arg_class.txt (nice if we can export as latex)