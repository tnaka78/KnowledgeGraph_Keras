import os
from collections import defaultdict
import numpy as np
import timeit

class KnowledgeGraph:

    @property
    def num_triple_train(self):
        return self.__num_triple_train

    @property
    def num_triple_test(self):
        return self.__num_triple_test

    @property
    def num_entity(self):
        return self.__num_entity

    @property
    def num_relation(self):
        return self.__num_relation

    @property
    def hr_t(self):
        return self.__hr_t

    @property
    def tr_h(self):
        return self.__tr_h

    def __init__(self, data_dir, negative_sampling):
        self.__data_dir = data_dir
        self.__negative_sampling = negative_sampling

        self.__entity2id = {}
        self.__id2entity = {}
        self.__relation2id = {}
        self.__id2relation = {}
        self.__hr_t = defaultdict(set)
        self.__tr_h = defaultdict(set)

        self.__triple_train = []
        self.__triple_test = []
        self.__triple_valid = []

        self.__num_entity = 0
        self.__num_relation = 0
        self.__num_triple_train = 0
        self.__num_triple_test = 0
        self.__num_triple_valid = 0

        self.__load_data()
        print('finish preparing data. ')

    def get_training_data(self):
        n_triple = len(self.__triple_train)
        triple_set = set(self.__triple_train)
        rand_idx = np.random.permutation(n_triple)
        train_triple_positive = np.asarray([self.__triple_train[x] for x in rand_idx])
        train_triple_negative = []
        for t in train_triple_positive:
            if self.__negative_sampling == 'unif':
                replace_head_probability = 0.5
            elif self.__negative_sampling == 'bern':
                replace_head_probability = self.__relation_property[t[1]]
            else:
                raise Exception("Invalid negative_sampling : {}".format(self.__negative_sampling))

            while True:
                replace_entity_id = np.random.randint(self.__num_entity)
                random_num = np.random.random()

                if random_num < replace_head_probability:
                    triple = (replace_entity_id, t[1], t[2])
                else:
                    triple = (t[0], t[1], replace_entity_id)

                if triple not in triple_set:
                    train_triple_negative.append(list(triple))
                    break

        train_triple_negative = np.asarray(train_triple_negative)

        return train_triple_positive, train_triple_negative

    def get_test_data(self):
        return self.__triple_test

    def __load_triple(self, filename):
        triples = list()
        with open(os.path.join(self.__data_dir, filename)) as f:
            for line in f.readlines():
                elements = line.strip().split('\t')
                assert len(elements) == 3
                h = self.__entity2id[elements[0]]
                r = self.__relation2id[elements[2]]
                t = self.__entity2id[elements[1]]
                triples.append((h, r, t))
                self.__hr_t[(h, r)].add(t)
                self.__tr_h[(t, r)].add(h)
        return triples

    def __load_iddef(self, filename):
        name2id = dict()
        id2name = dict()
        with open(os.path.join(self.__data_dir, filename)) as f:
            for line in f.readlines():
                elements = line.strip().split('\t')
                assert len(elements) == 2
                name = elements[0]
                id = int(elements[1])
                name2id[name] = id
                id2name[id] = name
        return name2id, id2name

    def __load_data(self):
        self.__entity2id, self.__id2entity = self.__load_iddef('entity2id.txt')
        self.__relation2id, self.__id2relation = self.__load_iddef('relation2id.txt')

        self.__triple_train = self.__load_triple('train.txt')
        self.__triple_test = self.__load_triple('test.txt')
        self.__triple_valid = self.__load_triple('valid.txt')
        self.__triple = np.concatenate([self.__triple_train, self.__triple_test, self.__triple_valid], axis=0)

        self.__num_relation = len(self.__relation2id)
        self.__num_entity = len(self.__entity2id)
        self.__num_triple_train = len(self.__triple_train)
        self.__num_triple_test = len(self.__triple_test)
        self.__num_triple_valid = len(self.__triple_valid)

        if self.__negative_sampling == 'bern':
            self.__relation_property_head = {x: [] for x in
                                             range(self.__num_relation)}
            self.__relation_property_tail = {x: [] for x in
                                             range(self.__num_relation)}
            for (h, r, t) in self.__triple_train:
                self.__relation_property_head[r].append(h)
                self.__relation_property_tail[r].append(t)
            self.__relation_property = dict()
            for r in self.__relation_property_head.keys():
                num_heads = len(set(self.__relation_property_head[r]))
                num_tails = (len(set(self.__relation_property_tail[r])))
                ratio = num_tails / (num_heads + num_tails)
                self.__relation_property[r] = ratio
