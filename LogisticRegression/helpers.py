

def didi_este_un_purcel():
    print("")
    print("+=============+")
    print("|Groh, groh...|")
    print("+=============+")


import random
from random import shuffle
import json
import numpy as np

def random_split(seed, data):
    # This is a way to go about doing random splits without Sframe,
    # but it won't be the same like random_split in Sframe even with the same seed

    random.seed(seed)
    train_data_size = int(len(data) * .8)
    indices = range(len(data))
    shuffle(indices)
    train_data_rnd = data.iloc[indices[:train_data_size]]
    test_data_rnd = data.iloc[indices[train_data_size:]]
    #print len(test_data_rnd)
    #print len(train_data_rnd)
    #print len(products)
    return train_data_rnd, test_data_rnd

def shuffle(array):
    copy = list(array)
    shuffle_in_place(copy)
    return copy

def shuffle_in_place(array):
    # example of shuffle implementation
    array_len = len(array)
    assert array_len > 2, 'Array is too short to shuffle!'
    for index in range(array_len):
        swap = random.randrange(array_len - 1)
        swap += swap >= index
        array[index], array[swap] = array[swap], array[index]

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

def get_data_from_json_indexes(file, data_source):
    with open(file) as data_file:
        test_idx = json.load(data_file)
    return data_source.iloc[test_idx]

def get_probability(score):
    return 1/(1+np.exp(-score))

def get_accuracy(data, target_column, probabilities_column):
    true_positives = len(data[(data[target_column] == 1) & (data[probabilities_column] > 0.5)])
    # print true_positives
    true_negatives = len(data[(data[target_column] == -1) & (data[probabilities_column] <= 0.5)])
    # print true_negatives

    accuracy = (true_positives + true_negatives) / len(data)
    # print "accuracy: %.2f" % accuracy
    return accuracy