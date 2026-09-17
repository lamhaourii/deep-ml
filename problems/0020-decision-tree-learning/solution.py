import math
from collections import Counter
import numpy as np

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    labels= np.array(labels)
    _, counts= np.unique(labels,return_counts=True )
    freqs= counts/np.size(labels)
    return -np.sum(freqs*np.log(freqs))


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    target_labels= [example[target_attr] for example in examples] 
    entropy= calculate_entropy(target_labels)  
    ctr= Counter([example[attr] for example in examples])
    labels=[[example[target_attr] for example in examples if example[attr]==value] for value in ctr.keys()]
    entropies=np.array([calculate_entropy(lab) for lab in labels])
    freqs= np.array(list(ctr.values()))/len(examples)  
    return entropy - np.sum(freqs*entropies)

    

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    ctr= Counter([example[target_attr] for example in examples])
    ctr= sorted(ctr.items(), key=lambda pair: (-pair[1], pair[0]))
    maj= ctr[0][0]
    return maj

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    if not attributes:
        return  majority_class(examples,target_attr)
    if len(set([example[target_attr] for example in examples]))==1:
        return examples[0][target_attr]
    max_gain=float('-inf')
    
    for attr in attributes:
        gain= calculate_information_gain(examples, attr,target_attr)
        if gain>max_gain:
            max_gain, att=gain, attr
    att_uniques= sorted(set([example[att] for example in examples]))
    #divided=[[example for example in examples if example[att]==val] for val in att_uniques]
    new_attr= attributes.copy()
    new_attr.remove(att)
    return {
        att: {val:learn_decision_tree([example for example in examples if example[att]==val], new_attr, target_attr)
        for val in att_uniques}
    }




    