# all functions for feature extraction
import csv
from collections import defaultdict
import networkx as nx

# function 'shortest_dependency_path' inspired by: 
#    and:
#   
# adapted to work with gold data instead of spacy

def shortest_dependency_path(tokens, children, dep_rel, e1=None, e2=None):
    """ Returns a string representing the shortest path between 
            two tokens, shown by dependency
    Args:
        tokens: a conllu of the feature extractions including gold labels
        children:
        dep_rel:
        e1:
        e2:
    
    Returns:
        dep_path:
    """
    edges = []
    relations=defaultdict()
    for token in tokens:
        
        for i, child in enumerate(children[token]):
            edges.append(('{0}'.format(token),
                          '{0}'.format(child)))
            relations[(token, child)] = dep_rel[token][i]
            
    graph = nx.Graph(edges)
    
    try:
        shortest_path = nx.shortest_path(graph, source=e1, target=e2)
    except nx.NetworkXNoPath:
        shortest_path = []

    dep_path=['+']
    for i in range(len(shortest_path)-1):
        tupl = (shortest_path[i], shortest_path[i+1])
        tupl2 = (shortest_path[i+1], shortest_path[i])
        
        # vary in the direction of the relation via "u(up) or d(down)"
        try:
            dep_path.append('u'+relations[tupl])
        except:
            dep_path.append('d'+relations[tupl2])

    dep_path = ''.join(word for word in dep_path)

    return dep_path



def get_sents_and_preds(data):
    """
    
    """
    all_sentences=[]
    sentence_tokens=[]
    all_predicates=[]
    
    # get a list of sentence tokens, and a list of predicates per sentences     
    with open(data, 'r', encoding='utf-8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='"')

        # get all tokens in sentence and predicate of sentence
        for token in file:
            if token == []:
                all_predicates.append(predicate)
                predicate='_'
                all_sentences.append(sentence_tokens)
                sentence_tokens=[]
                continue
            
            elif token[0].startswith('#') or token[0].startswith('"'):
                continue    
            
            else:
                sentence_tokens.append(token[1])
                if token[10] != '_':
                    predicate=token[1]

    return all_sentences, all_predicates


def get_child_tokens(data):   
    """
    
    """
    all_sentences, all_predicates = get_sents_and_preds(data)
    children_of_tokens = defaultdict(lambda: defaultdict(list))
    dep_rel_children = defaultdict(lambda: defaultdict(list))
    
    # get a list of sentence tokens, and a list of predicates per sentences     
    with open(data, 'r', encoding='utf-8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='"')

        # get children of each token in dict, with the relation of each child to head    
        sentence_num = 0
        for token in file:
            if token == []:
                sentence_num+=1
                continue

            elif token[0].startswith('#') or token[0].startswith('"'):
                    continue 

            else:
                child_of_ind = token[6]
                child_of_token = all_sentences[sentence_num][int(child_of_ind)-1]
                children_of_tokens[sentence_num][child_of_token].append(token[1])
                dep_rel_children[sentence_num][child_of_token].append(token[7])
    
    return children_of_tokens, dep_rel_children, all_predicates, all_sentences

def extract_features_and_labels(inputfile, data_type):
    """
    inputfile: preprocessed conllu file

    """
    children_of_tokens, dep_rel_children,\
        all_predicates, all_sentences = get_child_tokens(inputfile)

    with open(inputfile, encoding='utf-8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='"')
        
        next_tokens =[]
        next_poss = []
        for row in file:
            # if current row is empty, it marks a sentence boundary
            if row == []:
                next_tokens.append('_')
                next_poss.append('_')
            
            # if token is a predicate: append to list of predicates of current sentence
            elif row[0].startswith('#') or row[0].startswith('"'):
                next_tokens.append('_')
                next_poss.append('_')
                    
            else: 
                next_tokens.append(row[1])
                next_poss.append(row[3])
        
        next_tokens.append('_')
        next_tokens.pop(0)
        next_poss.append('_')
        next_poss.pop(0)

    with open(inputfile, encoding='utf-8') as file:
        file = csv.reader(file, delimiter='\t', quotechar='"')

        output_filepath = 'output/'+data_type+'features.conllu'
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            output = csv.writer(csvfile, delimiter='\t')
            features = []
            labels = []
            
            prev_token = '_'
            prev_pos = '_'
            
            current_sent = 0
            for i, row in enumerate(file):
                if row == []:
                    prev_token = '_'
                    prev_pos = '_'
                    output.writerow(row)
                    current_sent+=1
                
                elif row[0].startswith('#') or row[0].startswith('"'):
                    output.writerow(row)
                    continue
                else:
                    # set variables for all features
                    token_features = {}
                    
                    # current and next, token and pos
                    current_token = row[1]
                    current_pos = row[3]
                    next_token = next_tokens[i]
                    next_pos = next_poss[i]

                    sentence = all_sentences[current_sent]
                    children = children_of_tokens[current_sent]
                    dep_rels = dep_rel_children[current_sent]
                    

                    # 
                    if row[6] > row[0]: # if current is before head
                        head_init = 0
                    elif row[6] < row[0]:
                        head_init = 1
                    else:
                        head_init = 2
                    
                    # find the head predicate and relation to head
                    head_ind = int(row[6])-1
                    head = all_sentences[current_sent][head_ind]
                    predicate = all_predicates[current_sent]
                    
                    dependencypath = shortest_dependency_path(sentence, children,dep_rels, row[1], predicate)
                    
                    token_features = {'token':current_token, 'pos':current_pos, 
                            'prev_token':prev_token, 'prev_pos':prev_pos, 
                            'next_token': next_token, 'next_pos': next_pos,
                            'head':head, 'head_init':head_init, 'predicate': predicate,
                            'dependency_path':dependencypath}
                    
                    new_row = [current_token, current_pos, prev_token, prev_pos, next_token, 
                                next_pos, head, head_init, predicate, dependencypath]
                    
                    new_row.append(row[-2])
                    new_row.append(row[-1])
                    
                    output.writerow(new_row)
                    # reassign prev token after adding to feature list
                    prev_token = row[1]
                    prev_pos = row[3]
                
                    labels.append(row[11])
                    features.append(token_features)
    
    
    if data_type == 'train':
        return features, labels
    else:
        return features
