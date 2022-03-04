import argparse

def define_commandline_input():
    '''
        Defines arguments. options and commands for the commandline (using the argparse package)
        @rtype: ArgumentParser
        @returns: parser with all commandline arguments, options and commands
    '''
    
    #https://docs.python.org/3/library/argparse.html
    
    parser = argparse.ArgumentParser(description='Specify which step to run in file')
    
    # arguments that run parts of the code
    parser.add_argument('-preproc','--preprocess_files', action='store_true', help='Preprocess the files and extract features')
    parser.add_argument('-pred_ex','--predicate_extract', action='store_true', help='Extract predicates, rule-based')
    parser.add_argument('-arg_ex','--argument_extract', action='store_true', help='Extract arguments, rule-based')
    parser.add_argument('-arg_class','--argument_classification', action='store_true', help='Classify arguments, logistic regression')
    parser.add_argument('-all','--run_all_steps', action='store_true', help='preprocess, extract preds and arg, classify arguments')
    
    return parser

def check_arguments(my_arguments):
    '''
        Function that checks if specifications make sense
    '''
    if not my_arguments.preprocess_files and not my_arguments.predicate_extract \
    and not my_arguments.argument_extract and not my_arguments.argument_classification \
    and not my_arguments.run_all_steps:

        print("Please specify which step you would like to run:\n '-preproc' \n '-pred_ex' \n '-arg_ex' \n '-arg_class' \n '-all' ")
    
