from util import preprocess_gold, preprocess_argclass, \
                extract_features_and_labels,\
                gold_predicate_labels, extract_predicates, \
                allowed_dep_paths, write_arg_prediction, \
                argument_extraction, get_features_gold_labels,\
                create_classifier, classify_data, write_eval,\
                define_commandline_input, check_arguments

# edit if you want to run the tasks on the test data ('test')
data_type = 'dev'


def main():
    """ This main function is devided into 4 parts to preprocess files, 
    extract arguments and predicates and classify predicates. Each step is 
    based on the gold labels. THe output of one step is thus not the input 
    of the next.
      
    """
    parser = define_commandline_input()
    my_arguments = parser.parse_args()
    check_arguments(my_arguments)

    if my_arguments.preprocess_files or my_arguments.run_all_steps:
        ###################################
        ###    Preprocess datafiles     ### (duplicate sentences per predicate)
        ###################################
        data_files = ['data/en_ewt-up-train.conllu', 'data/en_ewt-up-dev.conllu', 'data/en_ewt-up-test.conllu']
        for data in data_files:
            preprocess_gold(data)
            preprocess_argclass(data)

        ###################################
        ###     Feature extraction      ###
        ###################################
        extract_features_and_labels('data/en_ewt-up-train_preprocessed.conllu', 'train')
        extract_features_and_labels('data/en_ewt-up-dev_preprocessed.conllu', 'dev')
        extract_features_and_labels('data/en_ewt-up-test_preprocessed.conllu', 'test')

    if my_arguments.predicate_extract or my_arguments.run_all_steps:
        ###################################
        ###     Predicate extraction    ###
        ###################################
        # Step 1: with original file, find all 'VERB' in dev/test
        extracted_predicates = extract_predicates('data/en_ewt-up-'+data_type+'.conllu', 'dev')
        gold_predicates = gold_predicate_labels('data/en_ewt-up-'+data_type+'.conllu')

        ## Step 2: Evaluate extraction
        write_eval(gold_predicates, extracted_predicates, 'pred_extract', data_type)

    if my_arguments.argument_extract or my_arguments.run_all_steps:
        ###################################
        ###     Argument extraction     ###
        ###################################
        gold_path = 'data/en_ewt-up-'+data_type+'_preprocessed.conllu'
        machine_path = 'output/'+data_type+'_extract_arguments.conllu'
        ## Step 1: Prepare set of allowed dependecy paths 
        allowed_paths = allowed_dep_paths('output/trainfeatures.conllu')

        ## Step 2: extract and write argument dev/test
        write_arg_prediction(gold_path, 'output/'+data_type+'features.conllu', 
                                allowed_paths, data_type=data_type)

        # Step 3: get labels from files
        extracted_args, gold_args = argument_extraction(gold_path,machine_path)

        ## Step 4: Evaluate extraction
        write_eval(gold_args, extracted_args, 'arg_extract', data_type)

    if my_arguments.argument_classification or my_arguments.run_all_steps:
        #################################
        ###    Classify Arguments     ###
        #################################
        ## Step 1: extract features (use feature data)
        train_features, train_labels = get_features_gold_labels('train')

        ## Step 2: Train classifier
        model, vec = create_classifier(train_features, train_labels)

        ## Step 3: Classify new data
        machine_features, gold_machine_labels = get_features_gold_labels(data_type)
        predictions = classify_data(model, machine_features, vec=vec)

        ## Step 4: evaluate classifier:
        write_eval(gold_machine_labels, predictions, 'arg_class', data_type)

if __name__ == '__main__':
    main()