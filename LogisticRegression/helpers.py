def didi_este_un_purcel():
    print("")
    print("+=============+")
    print("|Groh, groh...|")
    print("+=============+")


import random
from random import shuffle

def random_split(seed, data):
    # This is a way to go without Sframe,
    # but it won't be the same like random_split in Sframe even with the same seed
    # so I will have to use the data from json
    # wondering how shuffle is implemented

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