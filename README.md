# NLP Technologies - Assignment 2 

## General Info
This code extract the predicates and arguments from a sentence. 
It does each step seperately, and uses the Universal Propbank ConLL data for each step.
It finds the predicates by simply selecting all verbs. To find the arguments it uses the predicates as stated in the original data.
It does not use the output of step 1 as input for step 2.
For step 3 the code create a classifier to find the argument type for each argument in the data.
The classifier is based on a Support Vector Machine.


### How to run
Running *main.py* from commandline is done with various arguments:
`pred_extract`: only extract predicates, stores the output in the output folder
`arg_extract`: only does argument extraction, also stores the output
`arg_classify`: only classifies the arguments
`full`: Runs all steps in order of the arguments above

### Features for classification

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



### Code structure
 
- *main.py*
- folder *util* : 
	- *preprocess_file_structure.py* : contains the function to restructure the conll file
	- *pred_extract.py* : contains all functions for predicate extraction.
	- *arg_extract.py* : contains all functions for argument extraction
	- *feat_extract.py* : contains all functions to extract the features needed for argument classification
	- *arg_classifier.py* : contains all functions to create and run the classifier
	- *evaluation.py* : contains evaluation metrics functions

- folder *data* :  Contains the conll data sets (cite)
- folder *output*: contains all results from the extractions and classifications
- 

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
