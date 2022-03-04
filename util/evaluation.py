# All functions for evaluation
from sklearn.metrics import classification_report
import pandas as pd

def write_eval(gold, machine, task, data_type):
    """Stores the evaluation of the needed evaluation as latex
    Args:
        gold: a list of gold labels
        machine: a list of extracted/machine labels
        task: a string to show the task: 'pred_extract', 'arg_extract', 'arg_class'
        data_type: a string representing data type: 'dev', 'test'
    
    Returns:
        -
    """

    outpath = 'output/'+ task + '_'+ data_type +'_eval.txt'
    argument_eval = classification_report(y_true=gold,
                                            y_pred=machine,
                                            output_dict=True)

    arg_eval_df = pd.DataFrame(argument_eval).T
    arg_eval_df.to_latex(outpath)
    
