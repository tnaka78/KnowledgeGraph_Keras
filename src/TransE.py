import math
import argparse
import keras as K
import numpy as np
from KnowledgeGraph import KnowledgeGraph
from common import *


class TransE:
    @property
    def embedding_entity(self):
        return self.__embedding_entity

    @property
    def embedding_relation(self):
        return self.__embedding_relation

    def __init__(self, num_entity, num_relation, learning_rate, batch_size, num_epochs, margin, dimension):
        self.__num_entity = num_entity
        self.__num_relation = num_relation
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__num_epochs = num_epochs
        self.__margin = margin
        self.__dimension = dimension

        bound = 6 / math.sqrt(self.__dimension)
        self.__embedding_entity = K.layers.Embedding(self.__num_entity, self.__dimension, name='embedding_entity',
                                                     embeddings_initializer=K.initializers.random_uniform(minval=-bound, maxval=bound))
        self.__embedding_relation = K.layers.Embedding(self.__num_relation, self.__dimension, name='embedding_relation',
                                                       embeddings_initializer=K.initializers.random_uniform(minval=-bound, maxval=bound))

        self.__train_model = None
        self.__predict_model = None
        self.__test_model = None

    def __compile_train_model(self):
        positive_head = K.Input((1,), dtype='int32', name='positive_heads')
        positive_relation = K.Input((1,), dtype='int32', name='positive_relations')
        positive_tail = K.Input((1,), dtype='int32', name='positive_tails')
        negative_head = K.Input((1,), dtype='int32', name='negative_heads')
        negative_tail = K.Input((1,), dtype='int32', name='negative_tails')

        embedding_positive_head = self.__embedding_entity(positive_head)
        embedding_positive_tail = self.__embedding_entity(positive_tail)
        embedding_positive_relation = self.__embedding_relation(positive_relation)
        embedding_negative_head = self.__embedding_entity(negative_head)
        embedding_negative_tail = self.__embedding_entity(negative_tail)

        embedding_positive_triple = [embedding_positive_head, embedding_positive_relation, embedding_positive_tail]
        embedding_negative_triple = [embedding_negative_head, embedding_positive_relation, embedding_negative_tail]
        score_positive = K.layers.Lambda(lambda x: K.backend.sqrt(K.backend.sum(K.backend.square(x[0] + x[1] - x[2]), axis=1)+1e-10))(embedding_positive_triple)
        score_negative = K.layers.Lambda(lambda x: K.backend.sqrt(K.backend.sum(K.backend.square(x[0] + x[1] - x[2]), axis=1)+1e-10))(embedding_negative_triple)
        loss = K.layers.Lambda(lambda x: K.backend.maximum(x[0] + self.__margin - x[1], 0.0))([score_positive, score_negative])

        self.__train_model = K.Model(inputs=[positive_head, positive_relation, positive_tail, negative_head, negative_tail], outputs=loss)
        opt = K.optimizers.Adam(lr=self.__learning_rate)
        self.__train_model.compile(opt, loss=lambda y_true, y_pred: y_pred)

    def __compile_eval_model(self):
        head = K.Input((1,), dtype='int32', name='heads')
        relation = K.Input((1,), dtype='int32', name='relations')
        tail = K.Input((1,), dtype='int32', name='tails')
        embedding_head = self.__embedding_entity(head)
        embedding_tail = self.__embedding_entity(tail)
        embedding_relation = self.__embedding_relation(relation)
        embedding_triple = [embedding_head, embedding_relation, embedding_tail]
        loss = K.layers.Lambda(lambda x: K.backend.sqrt(K.backend.sum(K.backend.square(x[0] + x[1] - x[2]), axis=1)))(embedding_triple)

        self.__predict_model = K.Model(inputs=[head, relation, tail], outputs=loss)
        opt = K.optimizers.Adam(lr=self.__learning_rate)
        self.__predict_model.compile(opt, loss='binary_crossentropy')

    def __compile_test_model(self):
        head = K.Input((self.__num_entity,), dtype='int32', name='heads')
        relation = K.Input((self.__num_entity,), dtype='int32', name='relations')
        tail = K.Input((self.__num_entity,), dtype='int32', name='tails')
        embedding_head = self.__embedding_entity(head)
        embedding_tail = self.__embedding_entity(tail)
        embedding_relation = self.__embedding_relation(relation)
        embedding_triple = [embedding_head, embedding_relation, embedding_tail]
        loss = K.layers.Lambda(lambda x: K.backend.sqrt(K.backend.sum(K.backend.square(x[0] + x[1] - x[2]), axis=2)))(embedding_triple)

        self.__test_model = K.Model(inputs=[head, relation, tail], outputs=loss)
        opt = K.optimizers.Adam(lr=self.__learning_rate)
        self.__test_model.compile(opt, loss='binary_crossentropy')

    def compile(self):
        self.__compile_train_model()
        self.__compile_eval_model()
        self.__compile_test_model()

    def train(self, ph, pr, pt, nh, nt):
        label = np.zeros((len(ph), 1))
        self.__train_model.fit(x=[ph, pr, pt, nh, nt], y=label, epochs=self.__num_epochs, batch_size=self.__batch_size)

    def evaluate(self, h, r, t):
        score = self.__predict_model.predict(x=[h, r, t])
        print(score)
        return score

    def predict_head(self, r, t):
        test_num = len(r)
        heads = np.tile(np.arange(0, self.__num_entity), [test_num, 1])
        relations = np.tile(np.reshape(r, [test_num, 1]), [1, self.__num_entity])
        tails = np.tile(np.reshape(t, [test_num, 1]), [1, self.__num_entity])
        score = self.__test_model.predict(x=[heads, relations, tails])
        return score

    def predict_tail(self, h, r):
        test_num = len(r)
        heads = np.tile(np.reshape(h, [test_num, 1]), [1, self.__num_entity])
        relations = np.tile(np.reshape(r, [test_num, 1]), [1, self.__num_entity])
        tails = np.tile(np.arange(0, self.__num_entity), [test_num, 1])
        score = self.__test_model.predict(x=[heads, relations, tails])
        return score

    def save_embeddings(self):
        w_entity = self.__embedding_entity.get_weights()
        np.savetxt('./entity.tsv', w_entity[0], delimiter='\t')
        w_relation = self.__embedding_relation.get_weights()
        np.savetxt('./relation.tsv', w_relation[0], delimiter='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TransE")
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='../data/FB15k/')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4096)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=100)
    parser.add_argument('--dimension', dest='dimension', type=int, default=50)
    parser.add_argument('--margin', dest='margin', type=float, help='margin', default=1.0)
    parser.add_argument('--negative_sampling', dest='negative_sampling', type=str,
                        help='choose unit or bern to generate negative examples', default='bern')
    args = parser.parse_args()
    print(args)
    KG = KnowledgeGraph(data_dir=args.data_dir, negative_sampling=args.negative_sampling)
    model = TransE(num_entity=KG.num_entity, num_relation=KG.num_relation, learning_rate=args.learning_rate,
                   batch_size=args.batch_size, num_epochs=args.num_epochs, margin=args.margin, dimension=args.dimension)
    model.compile()
    tp, tn = KG.get_training_data()
    train_model(model, tp, tn)
    model.save_embeddings()
    test_model(model, KG.get_test_data())
