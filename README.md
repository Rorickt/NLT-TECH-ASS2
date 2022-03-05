# NLP Technologies - Assignment 2 

## General Info
This code extract the predicates and arguments from a sentence. 
It does each step seperately, and uses the Universal Propbank ConLL data for each step.
It finds the predicates by simply selecting all verbs. To find the arguments it uses the predicates as stated in the original data.
It does not use the output of step 1 as input for step 2.
For step 3 the code create a classifier to find the argument type for each argument in the data.
The classifier is based on a Logistic Regression.


## How to run

#### Assignment 2.1: 
Running *main.py* from commandline is done with various arguments:\
`-preproc`: Preprocess files for following codes (required to run once)\
`-pred_ex`: only extract predicates, stores the output in the output folder\
`-arg_ex`: only does argument extraction, also stores the output\
`-arg_class`: only classifies the arguments\
`-all`: Runs all steps in order of the arguments above (including preprocessing)\

#### Assignment 2.2: 
Run *main.py* from assignment 2.1 with the commandline argument `preproc` (unless already done). 
Following step 1, run *conll_to_jsonl.py* for the train data and for the dev data. 
Finally, with the files readied, run *srl_main.py*, which should print two labeled examples.

## Features for classification

- token itself
- POS-tag
- previous token
- previous POS-tag
- next token
- next POS-tag
- head
- head init
- predicate
- dependency path

## Code structure
 
- *main.py*
- *srl_main.py*
- *conll_to_json.py*
- folder *util* : 
	- *preprocess_file_structure.py* : contains the function to restructure the conll file
	- *pred_extract.py* : contains all functions for predicate extraction.
	- *arg_extract.py* : contains all functions for argument extraction
	- *feat_extract.py* : contains all functions to extract the features needed for argument classification
	- *arg_classifier.py* : contains all functions to create and run the classifier
	- *evaluation.py* : contains evaluation metrics functions
-folder *tools*	:
	- *srl_model_bert.py*
	- *srl_model_ltsm.py*
	- *srl_predictor.py*
	- *srl_reader.py*

- folder *data* :  Contains the conll data sets (cite)
- folder *output*: contains all results from the extractions and classifications

\
\
----CONTACT---- \
Mekselina Doğanç: m.doganc@student.vu.nl \
Mojca Kloos: m.c.kloos@student.vu.nl  \
Rorick Terlou: m.r.terlou@student.vu.nl \



**Needed modules and packages**

- Pandas
- Numpy
- argparse
- From sklearn: 
	- feature_extraction.DictVectorizer
	- classification_report
	- linear_model.LogisticRegression 
- From collections:
	- defaultdict
- from nx:
	- nx
	- shortest_path
- json
- tempfile
- from typing:
	- Dict
	- Iterable
	- List
	- Tuple
- allennlp
- torch
