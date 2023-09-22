####################################################
##### This code is for ML 6350 HW1 - Problem 2 #####
####################################################

#import libraries
from collections import Counter
import copy
import csv
import math
import os

#define function to calculate GI
def calc_GI(labels):
    labels_counter = Counter(labels)
    total = len(labels)
    gi = 1
    for label, count in labels_counter.items():
        p = float(count)/total
        p_sq = p*p
        gi -= p_sq
    return gi
# define function to split on attributes for calculating GI
def get_split_attr_GI(examples, attrs, labels):
    assert len(examples)==len(labels)

    
    #remove all logging.debug lines
    #
    gi_s = calc_GI(labels)
    
    total_cnt = len(labels)
    max_gain = -1
    split_attr = ''
    for i, attr in enumerate(attrs.items()):
        attr_name, attr_vals = attr
        
        gi_attr = gi_s
        for val in attr_vals:
            
            labels_v_attr = [labels[j] for j, ex in enumerate(examples) if ex[i] == val]
            gi_v = calc_GI(labels_v_attr)
            gi_v = (float(len(labels_v_attr))/ total_cnt)* gi_v
            gi_attr = gi_attr - gi_v
        if max_gain < gi_attr:
            max_gain = gi_attr
            split_attr = attr_name
    

    return split_attr
# function to calculate ME
def calc_ME(labels):
    labels_counter = Counter(labels)
    total = len(labels)
    cnt_major = labels_counter.most_common(1)[0][1] if total > 0 else 0
    me = float(total-cnt_major)/total if total > 0 else 0

    return me
# function to split using ME
def get_split_attr_ME(examples,attrs,labels):
    assert len(examples)==len(labels)

    
    me_s = calc_ME(labels)
    total_cnt = len(labels)
    max_gain = -1
    split_attr = ''
    for i, attr in enumerate(attrs.items()):
        attr_name, attr_vals = attr
        me_attr = me_s
        for val in attr_vals:
            labels_v_attr = [labels[j] for j, ex in enumerate(examples) if ex[i] == val]
            me_v = calc_ME(labels_v_attr)
            me_v = (float(len(labels_v_attr))/ total_cnt)* me_v
            me_attr = me_attr - me_v

        if max_gain < me_attr:
            max_gain = me_attr
            split_attr = attr_name

    return split_attr
# function to calculate information gain
def calc_IG(labels):
    labels_counter = Counter(labels)
    total = len(labels)
    entropy = 0
    for label, val in labels_counter.items():
        p = float(val)/total
        entropy_val = -1 * p * math.log2(p)
        entropy += entropy_val
    return entropy

# function to split on attribute with highest information gain
def get_split_attr_IG(examples,attrs,labels):
    assert len(examples)==len(labels)
    ig_s = calc_IG(labels)
    total_cnt = len(labels)
    max_gain = -1
    split_attr = ''
    for i, attr in enumerate(attrs.items()):
        attr_name, attr_vals = attr
        ig_attr = ig_s
        for val in attr_vals:
            labels_v_attr = [labels[j] for j, ex in enumerate(examples) if ex[i] == val]
            ig_v = calc_IG(labels_v_attr)
            ig_v = (float(len(labels_v_attr))/ total_cnt)* ig_v
            ig_attr = ig_attr - ig_v
        if max_gain < ig_attr:
            max_gain = ig_attr
            split_attr = attr_name
    

    return split_attr

#function to grab the best attribute to split on based on calculations
def get_best_split_attr(examples, attrs, labels, split_type):
    if split_type == "GI":
        split_attr = get_split_attr_GI(examples, attrs, labels)
    elif split_type == "ME":
        split_attr = get_split_attr_ME(examples, attrs, labels)
    elif split_type == "IG":
        split_attr = get_split_attr_IG(examples, attrs, labels)
    else:
        assert 1==0, "Wrong split type provided"
    
    return split_attr
# decision tree
def build_dtree(examples, attrs, labels, attr_order, depth=None, gain_type="IG"):
    # refernce https://docs.python.org/3/library/collections.html#collections.Counter
    most_common_label = Counter(labels).most_common(1)[0][0]
    if depth is not None and depth==0:
        return most_common_label
    if len(attrs) == 0:
        # most common label is returned
        return most_common_label
    if len(set(labels))==1:
        return labels[0]
    
    attr = get_best_split_attr(examples, attrs, labels, split_type=gain_type)
    attr_idx = attr_order.index(attr)
    tree = {attr : {}}
    for val in attrs[attr]:
        num_rows = [j for j, ex in enumerate(examples) if ex[attr_idx] == val]
        examples_v_attr = [examples[j] for j in num_rows]
        labels_v_attr = [labels[j] for j in num_rows]

        assert len(examples_v_attr)==len(labels_v_attr) 
        if len(examples_v_attr)==0:
            return most_common_label
        else:
            # Attributes - {A}
            v_attrs = copy.deepcopy(attrs)
            del v_attrs[attr]
            new_depth = None
            if depth is not None and isinstance(depth,int):
                new_depth = depth - 1
            subtree = build_dtree(examples_v_attr, v_attrs, labels_v_attr,attr_order, new_depth, gain_type)
            tree[attr][val] = subtree

    return tree
# grab label from tree
def get_label_from_dtree(tree, example, attrs, labels):
    assert len(example)==len(attrs.keys())


    unique_labels = set(labels)

    train_tree = tree
    if isinstance(train_tree, str) and train_tree in unique_labels:
        return train_tree
    while True:
        root_key = list(train_tree.keys())[0]
        value = example[list(attrs.keys()).index(root_key)]

        train_tree = train_tree[root_key][value]
        if isinstance(train_tree, str) and train_tree in unique_labels:
            return train_tree

def test_dtree(tree, test_examples, attrs, test_labels, unique_labels):
    assert tree is not None
    assert len(test_examples)==len(test_labels)

    accuracy = 0
    for ex, label in zip(test_examples, test_labels):
        predict_label = get_label_from_dtree(tree, ex, attrs,unique_labels)

        if predict_label==label:
            accuracy += 1
    return float(accuracy)/len(test_examples)

def q_2_2():
    csv_path = ["./cars-4/train.csv"]
    csv_path = os.path.join(*csv_path)

    attr_list = {"buying":['vhigh', 'high', 'med', 'low'],"maint": ['vhigh', 'high', 'med', 'low'],"doors": ["2", "3", "4", "5more"],
                 "persons": ['2', '4', 'more'],"lug_boot" : ['small', 'med', 'big'],"safety":['low', 'med', 'high']}
    examples = []
    labels = []
    with open(csv_path, 'r') as csv_file:
        train_reader = csv.reader(csv_file)
        for sample in train_reader:
            example, label = sample[: -1], sample[-1]
            examples.append(example)
            labels.append(label)
        
        assert len(labels)==len(examples)
    attr_order = list(attr_list.keys())
    tree_IG = build_dtree(examples, attr_list, labels,attr_order, None, 'IG')
    tree_ME = build_dtree(examples, attr_list, labels,attr_order, None, 'ME')
    tree_GI = build_dtree(examples, attr_list, labels,attr_order, None, 'GI')

    #adjusting the number affects acc. 
    trees = list()
    for i in range(1,7):
        trees.append(build_dtree(examples, attr_list, labels,attr_order, i, 'IG'))    
    
    unique_labels = ['unacc', 'acc', 'good', 'vgood']

    accuracy_IG = test_dtree(tree_IG, examples, attr_list, labels, unique_labels)
    print(f"Accuracy on training data for IG is {accuracy_IG}")

    accuracy_ME = test_dtree(tree_ME, examples, attr_list, labels, unique_labels)
    print(f"Accuracy on training data for ME is {accuracy_ME}")
    
    accuracy_GI = test_dtree(tree_GI, examples, attr_list, labels, unique_labels)
    print(f"Accuracy on training data for GI is {accuracy_GI}")

    for i, tree in enumerate(trees):
        accuracy = test_dtree(tree, examples, attr_list, labels, unique_labels)
        print(f"Accuracy on training data for tree with depth {i} is {accuracy}")
    
    csv_path = ["./cars-4/test.csv"]
    csv_path = os.path.abspath(*csv_path)
    test_examples = []
    test_labels = []
    with open(csv_path, 'r') as csv_file:
        train_reader = csv.reader(csv_file)
        for sample in train_reader:
            example, label = sample[: -1], sample[-1]
            test_examples.append(example)
            test_labels.append(label)
        
        assert len(test_labels)==len(test_examples)

    accuracy_IG = test_dtree(tree_IG, test_examples, attr_list, test_labels, unique_labels)
    print(f"Accuracy on test data for IG is {accuracy_IG}")    

    accuracy_ME = test_dtree(tree_ME, test_examples, attr_list, test_labels, unique_labels)
    print(f"Accuracy on test data for ME is {accuracy_ME}")
    
    accuracy_GI = test_dtree(tree_GI, test_examples, attr_list, test_labels, unique_labels)
    print(f"Accuracy on test data for GI is {accuracy_GI}")

    for i, tree in enumerate(trees):
        accuracy = test_dtree(tree, test_examples, attr_list, test_labels, unique_labels)
        print(f"Accuracy on training data for tree with depth {i+1} is {accuracy}")


def convert_numeric_to_binary(examples, numeric_attrs, attr_list):
    medians = []
    for attr in numeric_attrs:
        attr_indx = list(attr_list.keys()).index(attr)
        all_vals = [ex[attr_indx] for ex in examples]
        all_vals = sorted(all_vals)
        median_val = all_vals[len(examples)//2]
        medians.append((attr_indx, median_val))
    
    for ex in examples:
        for attr_indx, median in medians:
            ex[attr_indx] = "low" if ex[attr_indx] < median else "high"
    
    return examples
        
def fill_unknown_attrs(examples, attrs):
    possiblr_unkv_atts = ['job', "education", "contact", "poutcome"]
    replacements = []
    for attr in possiblr_unkv_atts:
        attr_indx = list(attrs.keys()).index(attr)
        all_vals = [ex[attr_indx] for ex in examples]
        most_common = Counter(all_vals).most_common(1)[0][0]
        replacements.append((attr_indx, most_common))

    for ex in examples:
        for attr_indx, replacement in replacements:
            ex[attr_indx] = replacement

    return examples

def q_2_3():
    csv_path = ("./bank-4/train.csv")
    csv_path = os.path.abspath(csv_path)

    attr_list = {"age":['low', 'high'], # originally numeric
                 "job": ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                        "blue-collar","self-employed","retired","technician","services"],
                 "marital": ["married","divorced","single"],
                 "education": ["unknown","secondary","primary","tertiary"],
                 "default" : ["yes","no"],
                 "balance":['low', 'high'], # originally numeric
                 "housing": ["yes","no"],
                 "loan": ["yes","no"],
                 "contact": ["unknown","telephone","cellular"],
                 "day": ["low", "high"], #originally numeric
                 "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                 "duration": ["low", "high"], # originally numeric
                 "campaign": ["low", "high"], # originally numeric
                 "pdays": ["low", "high"], # originally numeric
                 "previous": ["low", "high"], # originally numeric
                 "poutcome": [ "unknown","other","failure","success"],
                 }
    examples = []
    labels = []
    with open(csv_path, 'r') as csv_file:
        train_reader = csv.reader(csv_file)
        for sample in train_reader:
            example, label = sample[: -1], sample[-1]
            examples.append(example)
            labels.append(label)
        
        assert len(labels)==len(examples)

    numeric_attrs = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    examples = convert_numeric_to_binary(examples, numeric_attrs, attr_list)
    examples = fill_unknown_attrs(examples, attr_list)

    attr_order = list(attr_list.keys())
    tree = build_dtree(examples, attr_list, labels,attr_order, None)

    unique_labels = ["yes", "no"]

    accuracy = test_dtree(tree, examples, attr_list, labels, unique_labels)
    print(f"Accuracy on training data is {accuracy}")

    
    csv_path = ["./bank-4/test.csv"]
    csv_path = os.path.abspath(*csv_path)
    test_examples = []
    test_labels = []
    with open(csv_path, 'r') as csv_file:
        train_reader = csv.reader(csv_file)
        for sample in train_reader:
            example, label = sample[: -1], sample[-1]
            test_examples.append(example)
            test_labels.append(label)
        
        assert len(test_labels)==len(test_examples)

    test_examples = convert_numeric_to_binary(test_examples, numeric_attrs, attr_list)
    accuracy = test_dtree(tree, test_examples, attr_list, test_labels, unique_labels)
    print(f"Accuracy on test data is {accuracy}")    
    
    
if __name__=="__main__":
    q_2_2()
    q_2_3()
