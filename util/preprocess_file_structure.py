from collections import defaultdict
import csv

# what this does: get file with sentence entry for every predicate. If there are four predicates, then the sentence is added four times
# brainstormed with Dorien how to do this. Dorien made some changes to complete the code 

def preprocess_gold(data):
    """ Restructures original data to seperate (/duplicate) each sentence for each predicate
        Stores new data in data folder
    
    Args:
        data: string representing the filepath to the gold data
    """
    sentence_storage = defaultdict(lambda: defaultdict(list))
    sentence_num = 0
    tokens=[]
    predicates=[]

    ### reading the gold
    with open(data, 'r', encoding='utf-8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='§')
        for row in file:
            if row == []:
                
                sentence_storage[str(sentence_num)]['Predicates'] = predicates
                sentence_storage[str(sentence_num)]['Token_rows'] = tokens
                sentence_num+=1
                tokens=[]
                predicates=[]

            # ignore tokens that are a copy of another token
            elif len(row) > 2 and row[9].startswith('CopyOf'):
                pass
            #if row is not empty, append whole row to list tokens    
            else:
                tokens.append(row)
                if not row[0].startswith('#'):
                    if row[10] != '_': #if it is a predicate
                        predicates.append(row[10]) # store sense of predicate



    ### writing the new
    excepts_storage = []
    excepts = 0
    output_filepath = data.replace('.conllu', '_preprocessed.conllu')
    with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        output = csv.writer(csvfile, delimiter='\t')

        for sentence, values in sentence_storage.items():
            if values['Token_rows'][0][0].startswith('# propbank'):
                continue
            else:
                for i, predicate in enumerate(values['Predicates']):
                    for token in values['Token_rows']:
                        # write non-token rows as original
                        if token[0].startswith('#'):
                            output.writerow(token)
                        
                        # write token rows with only one predicate and the correlating column of arguments
                        else:
                            new_row = token[:10] # all columns up till sense info, excluding sense info
                            if token[10] == predicate:
                                new_row.append(token[2]) # lemma, not sense
                            else:
                                new_row.append('_') # underscore for every other token that is not predicate
                            
                            new_row.append(token[11+i]) # arg info for predicate in question
                            output.writerow(new_row)

                    output.writerow([])



def preprocess_argclass(data):
    """ Restructures original data to seperate (/duplicate) each sentence for each predicate
        Stores new data in data folder
    
    Args:
        data: string representing the filepath to the gold data
    """
    sentence_storage = defaultdict(lambda: defaultdict(list))
    sentence_num = 0
    tokens=[]
    predicates=[]

    ### reading the gold
    with open(data, 'r', encoding='utf-8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='§')
        for row in file:
            if row == []:
                
                sentence_storage[str(sentence_num)]['Predicates'] = predicates
                sentence_storage[str(sentence_num)]['Token_rows'] = tokens
                sentence_num+=1
                tokens=[]
                predicates=[]

            # ignore tokens that are a copy of another token
            elif len(row) > 2 and row[9].startswith('CopyOf'):
                pass
            #if row is not empty, append whole row to list tokens    
            else:
                tokens.append(row)
                if not row[0].startswith('#'):
                    if row[10] != '_': # if it is a predicate
                        predicates.append(row[10]) # store sense of predicate



    ### writing the new structure
    excepts_storage = []
    excepts = 0
    output_filepath = data.replace('.conllu', '_preprocessed_argclass.conllu')
    with open(output_filepath, 'w', newline='', encoding='utf8') as csvfile:
        output = csv.writer(csvfile, delimiter='\t')

        for sentence, values in sentence_storage.items():
            # ignore propbank lines as they do not contain arg-pred info
            if values['Token_rows'][0][0].startswith('# propbank'):
                continue
            else:
                for i, predicate in enumerate(values['Predicates']):
                    for token in values['Token_rows']:
                        # write original row when it isnt a token
                        if token[0].startswith('#'):
                            output.writerow(token)
                        # 
                        else:
                            new_row = token[:10] # all columns up till sense info, excluding sense info
                            if token[10] == predicate:
                                new_row.append(token[2]) #lemma, not sense
                            else:
                                new_row.append('_') #underscore for every other token that is not predicate

                            if token[11+i] == '_':
                                new_row.append('_') 
                            else:
                                new_row.append('ARG') # do not take arg type, only presence
                            output.writerow(new_row)
    
                    output.writerow([])