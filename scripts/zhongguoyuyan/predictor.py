#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
方言字音编码器.
'''

__author__ = '黄艺华 <lernanto@foxmail.com>'


import sys
import os
import logging
import datetime
import pandas
import numpy
import tensorflow as tf
from util import clean_data


class Predictor(tf.train.Checkpoint):
    '''预测指定方言点的字音.'''

    def __init__(
        self,
        dialects,
        chars,
        targets,
        dialect_emb_size=20,
        char_emb_size=20,
        target_sim='inner_product',
        target_bias=True,
        l2=0,
        optimizer=None,
        name='predictor'
    ):
        super().__init__(epoch=tf.Variable(0, dtype=tf.int64))

        self.dialects = tf.convert_to_tensor(dialects)
        self.chars = tf.convert_to_tensor(chars)
        self.targets = [tf.convert_to_tensor(t) for t in targets]
        self.dialect_emb_size = dialect_emb_size
        self.char_emb_size = char_emb_size
        self.target_sim = target_sim
        self.l2 = l2
        self.name = name
        self.variables = []

        self.dialect_table = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.dialects,
                values=tf.range(dialects.shape[0], dtype=tf.int64)
            ),
            num_oov_buckets=1
        )
        self.char_table = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.chars,
                values=tf.range(chars.shape[0], dtype=tf.int64)
            ),
            num_oov_buckets=1
        )

        init = tf.random_normal_initializer()

        self.dialect_emb = tf.Variable(init(
            shape=(self.dialect_table.size(), self.dialect_emb_size),
            dtype=tf.float32
        ), name='dialect_emb')
        self.char_emb = tf.Variable(init(
            shape=(self.char_table.size(), self.char_emb_size),
            dtype=tf.float32
        ), name='char_emb')

        self.target_tables = []
        for target in self.targets:
            self.target_tables.append(tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=target,
                    values=tf.range(target.shape[0], dtype=tf.int64)
                ),
                num_oov_buckets=1
            ))

        self.target_embs = []
        for i, target in enumerate(self.targets):
            self.target_embs.append(tf.Variable(init(
                shape=(target.shape[0], self.char_emb_size),
                dtype=tf.float32
            ), name=f'target_emb{i}'))

        self.add_variable(self.dialect_emb, self.char_emb, *self.target_embs)

        if self.target_sim == 'inner_product' and target_bias:
            self.target_biases = []
            for i, target in enumerate(self.targets):
                self.target_biases.append(tf.Variable(
                    init(shape=(target.shape[0],), dtype=tf.float32),
                    name=f'target_bias{i}'
                ))

            self.add_variable(*self.target_biases)

        self.optimizer = tf.optimizers.Adam(0.02) if optimizer is None else optimizer

    def add_variable(self, *args):
        self.variables.extend(args)

    def dialect_to_id(self, dialect):
        return self.dialect_table.lookup(tf.convert_to_tensor(dialect))

    def id_to_dialect(self, dialect_id):
        return tf.gather(self.dialects, dialect_id)

    def char_to_id(self, char):
        return self.char_table.lookup(tf.convert_to_tensor(char))

    def id_to_char(self, char_id):
        return tf.gather(self.chars, char_id)

    def target_to_id(self, index, target):
        return self.target_tables[index].lookup(tf.convert_to_tensor(target))

    def id_to_target(self, index, target_id):
        return tf.gather(self.targets[index], target_id)

    def get_dialect_emb(self, dialect):
        return tf.nn.embedding_lookup(self.dialect_emb, self.dialect_to_id(dialect))

    def get_char_emb(self, char):
        return tf.nn.embedding_lookup(self.char_emb, self.char_to_id(char))

    def get_target_emb(self, index, target):
        return tf.nn.embedding_lookup(
            self.target_embs[index],
            self.target_to_id(index, target)
        )

    def logits(self, dialect_emb, char_emb):
        emb = self.transform(dialect_emb, char_emb)

        if self.target_sim == 'inner_product':
            logits = [tf.matmul(emb, e, transpose_b=True) for e in self.target_embs]
            if hasattr(self, 'target_biases'):
                logits = [l + b[None, :] for l, b in zip(logits, self.target_biases)]

        elif self.target_sim == 'euclidean_distance':
            logits = [-tf.reduce_sum(
                tf.square(emb[:, None] - e[None, :]),
                axis=-1
            ) for e in self.target_embs]

        return logits

    def predict_id(self, dialect, char):
        dialect_emb = self.get_dialect_emb(dialect)
        char_emb = self.get_char_emb(char)
        logits = self.logits(dialect_emb, char_emb)
        return tf.stack(
            [tf.argmax(l, axis=1, output_type=tf.int32) for l in logits],
            axis=1
        )

    @tf.function
    def predict(self, dialect, char):
        ids = self.predict_id(dialect, char)
        return tf.stack(
            [self.id_to_target(i, ids[:, i]) for i in range(ids.shape[1])],
            axis=1
        )

    @tf.function
    def predict_proba(self, dialect, char):
        dialect_emb = self.get_dialect_emb(dialect)
        char_emb = self.get_char_emb(char)
        logits = self.logits(dialect_emb, char_emb)
        return [tf.nn.softmax(l) for l in logits]

    def predict_id_emb(self, dialect_emb, char_emb):
        logits = self.logits(dialect_emb, char_emb)
        return tf.stack(
            [tf.argmax(l, axis=1, output_type=tf.int32) for l in logits],
            axis=1
        )

    def predict_emb(self, dialect_emb, char_emb):
        ids = self.predict_id_emb(dialect_emb, char_emb)
        return tf.stack(
            [self.id_to_target(i, ids[:, i]) for i in range(ids.shape[1])],
            axis=1
        )

    def predict_proba_emb(self, dialect_emb, char_emb):
        logits = self.logits(dialect_emb, char_emb)
        return [tf.nn.softmax(l) for l in logits]

    @tf.function
    def loss(self, dialect, char, targets):
        dialect_emb = self.get_dialect_emb(dialect)
        char_emb = self.get_char_emb(char)
        logits = self.logits(dialect_emb, char_emb)

        target_ids = tf.stack(
            [self.target_to_id(i, targets[:, i]) for i in range(targets.shape[1])],
            axis=1
        )

        pred_ids = tf.stack(
            [tf.argmax(l, axis=1) for l in logits],
            axis=1
        )

        loss = []
        for i, (e, l) in enumerate(zip(self.target_embs, logits)):
            loss.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.minimum(target_ids[:, i], e.shape[0] - 1),
                logits=l
            ))
        loss = tf.stack(loss, axis=1)

        acc = tf.cast(target_ids == pred_ids, tf.float32)
        return loss, acc

    @tf.function
    def update(self, dialect, char, targets):
        with tf.GradientTape() as tape:
            loss, acc = self.loss(dialect, char, targets)
            weight = tf.cast(targets != '', tf.float32)
            grad = tape.gradient(
                tf.reduce_mean(tf.reduce_sum(loss * weight, axis=1)),
                self.variables
            )

        self.optimizer.apply_gradients(zip(grad, self.variables))
        return loss, acc, weight

    def train(
        self,
        train_data,
        eval_data,
        epochs=20,
        batch_size=100,
        output_path=None
    ):
        if output_path is None:
            output_path = os.path.join(
                self.name,
                f'{datetime.datetime.now():%Y%m%d%H%M}'
            )

        train_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        train_acc = tf.keras.metrics.MeanTensor(dtype=tf.float32)
        eval_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        eval_acc = tf.keras.metrics.MeanTensor(dtype=tf.float32)

        train_writer = tf.summary.create_file_writer(os.path.join(output_path, 'train'))
        eval_writer = tf.summary.create_file_writer(os.path.join(output_path, 'eval'))
        manager = tf.train.CheckpointManager(
            self,
            os.path.join(output_path, 'checkpoints'),
            max_to_keep=epochs
        )

        while self.epoch < epochs:
            for dialect, char, targets in train_data.shuffle(10000).batch(batch_size):
                loss, acc, weight = self.update(dialect, char, targets)
                train_loss.update_state(loss, weight)
                train_acc.update_state(
                    tf.reduce_mean(acc, axis=0),
                    tf.reduce_sum(weight, axis=0)
                )

            for dialect, char, targets in eval_data.batch(batch_size):
                loss, acc = self.loss(dialect, char, targets)
                weight = tf.cast(targets != '', tf.float32)
                eval_loss.update_state(loss, weight)
                eval_acc.update_state(
                    tf.reduce_mean(acc, axis=0),
                    tf.reduce_sum(weight, axis=0)
                )

            with train_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=self.epoch)
                acc = train_acc.result()
                for i in range(acc.shape[0]):
                    tf.summary.scalar(f'accuracy{i}', acc[i], step=self.epoch)

            with eval_writer.as_default():
                tf.summary.scalar('loss', eval_loss.result(), step=self.epoch)
                acc = eval_acc.result()
                for i in range(acc.shape[0]):
                    tf.summary.scalar(f'accuracy{i}', acc[i], step=self.epoch)

            train_loss.reset_states()
            train_acc.reset_states()
            eval_loss.reset_states()
            eval_acc.reset_states()

            self.epoch.assign_add(1)
            manager.save()

class LinearPredictor(Predictor):
    '''方言 embedding 和字 embedding 直接相加.'''

    def __init__(self, *args, emb_size=20, name='linear_predictor', **kwargs):
        super().__init__(
            *args,
            dialect_emb_size=emb_size,
            char_emb_size=emb_size,
            name=name,
            **kwargs
        )

    def transform(self, dialect_emb, char_emb):
        return dialect_emb + char_emb

class MLPPredictor(Predictor):
    '''使用 MLP 作为字音变换.'''

    def __init__(
        self,
        *args,
        hidden_layer=2,
        hidden_size=100,
        activation=tf.nn.relu,
        name='mlp_predictor',
        **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)

        self.activation = activation

        init = tf.random_normal_initializer()
        self.weights = []
        self.biases = []
        for i in range(hidden_layer):
            input_shape = self.dialect_emb_size + self.char_emb_size if i == 0 else hidden_size
            output_shape = self.char_emb_size if i == hidden_layer - 1 else hidden_size

            self.weights.append(tf.Variable(
                init(shape=(input_shape, output_shape), dtype=tf.float32),
                name=f'weight{i}')
            )
            self.biases.append(tf.Variable(
                init(shape=(output_shape,), dtype=tf.float32),
                name=f'bias{i}'
            ))

        self.add_variable(*self.weights, *self.biases)

    def transform(self, dialect_emb, char_emb):
        x = tf.concat([dialect_emb, char_emb], axis=1)
        for w, b in zip(self.weights, self.biases):
            x = self.activation(tf.matmul(x, w) + b[None, :])

        return x

class ResidualPredictor(Predictor):
    '''预测方言字音变换的残差.'''

    def __init__(
        self,
        *args,
        hidden_size=100,
        hidden_bias=True,
        activation=tf.nn.relu,
        name='residual_predictor',
        **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)

        self.activation = activation

        init = tf.random_normal_initializer()
        self.weight0 = tf.Variable(init(
            shape=(self.dialect_emb_size + self.char_emb_size, hidden_size),
            dtype=tf.float32
        ), name='weight0')
        self.weight1 = tf.Variable(
            init(shape=(hidden_size, self.char_emb_size), dtype=tf.float32),
            name='weight1'
        )
        self.add_variable(self.weight0, self.weight1)

        if hidden_bias:
            self.bias = tf.Variable(
                init(shape=(hidden_size,), dtype=tf.float32),
                name='bias'
            )
            self.add_variable(self.bias)

    def transform(self, dialect_emb, char_emb):
        hidden = tf.matmul(
            tf.concat([dialect_emb, char_emb], axis=1),
            self.weight0
        )

        if hasattr(self, 'bias'):
            hidden += self.bias[None, :]

        return char_emb + tf.matmul(self.activation(hidden), self.weight1)

class AttentionPredictor(Predictor):
    def __init__(
        self,
        *args,
        transform_heads=1,
        transform_size=100,
        dialect_att_bias=True,
        dialect_att_activation=tf.nn.sigmoid,
        char_att_bias=True,
        char_att_activation=tf.nn.sigmoid,
        transform_bias=False,
        name='attention_predictor',
        **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)

        self.dialect_att_activation = dialect_att_activation
        self.char_att_activation = char_att_activation

        init = tf.random_normal_initializer()
        self.dialect_att_weight = tf.Variable(init(
            shape=(self.dialect_emb_size, transform_heads, transform_size),
            dtype=tf.float32
        ), name='dialect_att_weight')
        self.char_att_weight = tf.Variable(init(
            shape=(self.char_emb_size, transform_heads, transform_size),
            dtype=tf.float32
        ), name='char_att_weight')
        self.trans_weight = tf.Variable(init(
            shape=(transform_heads, transform_size, self.char_emb_size),
            dtype=tf.float32
        ), name='trans_weight')

        self.add_variable(
            self.dialect_att_weight,
            self.char_att_weight,
            self.trans_weight
        )

        if dialect_att_bias:
            self.dialect_att_bias = tf.Variable(
                init(shape=(transform_heads, transform_size), dtype=tf.float32),
                name='dialect_att_bias'
            )
            self.add_variable(self.dialect_att_bias)

        if char_att_bias:
            self.char_att_bias = tf.Variable(
                init(shape=(transform_heads, transform_size), dtype=tf.float32),
                name='char_att_bias'
            )
            self.add_variable(self.char_att_bias)

        if transform_bias:
            self.transform_bias = tf.Variable(
                init(shape=(self.char_emb_size,), dtype=tf.float32),
                name='transform_bias'
            )
            self.add_variable(self.transform_bias)

    def dialect_att(self, dialect_emb):
        att = tf.tensordot(dialect_emb, self.dialect_att_weight, [1, 0])
        if hasattr(self, 'dialect_att_bias'):
            att += self.dialect_att_bias[None, :, :]
        return self.dialect_att_activation(att)

    def char_att(self, char_emb):
        att = tf.tensordot(char_emb, self.char_att_weight, [1, 0])
        if hasattr(self, 'char_att_bias'):
            att += self.char_att_bias[None, :, :]
        return self.char_att_activation(att)

    def transform(self, dialect_emb, char_emb):
        att = self.dialect_att(dialect_emb) * self.char_att(char_emb)
        trans = tf.tensordot(att, self.trans_weight, [[1, 2], [0, 1]])
        if hasattr(self, 'transform_bias'):
            trans += self.transform_bias[None, :]
        return char_emb + trans

def load_data(prefix, ids, suffix='mb01dz.csv'):
    '''加载方言字音数据'''

    logging.info(f'loading {len(ids)} files ...')

    data = []
    for id in ids:
        fname = os.path.join(prefix, id + suffix)
        try:
            d = pandas.read_csv(
                fname,
                encoding='utf-8',
                dtype={'iid': numpy.int64, 'initial': str, 'finals': str, 'tone': str},
            )
            d['oid'] = id
            data.append(d)
        except Exception as e:
            logging.error(f'cannot load file {fname}: {e}')

    dialects = len(data)
    data = clean_data(pandas.concat(data, ignore_index=True), minfreq=2)
    data = data[(data['initial'] != '') | (data['finals'] != '') | (data['tone'] != '')]

    logging.info(f'done. loaded {dialects} dialects {data.shape[0]} records')
    return data

def benchmark(data):
    dialects = data['oid'].unique()
    chars = data['iid'].unique()
    initials = data.loc[data['initial'] != '', 'initial'].unique()
    finals = data.loc[data['finals'] != '', 'finals'].unique()
    tones = data.loc[data['tone'] != '', 'tone'].unique()

    dataset = tf.data.Dataset.from_tensor_slices((
        data['oid'],
        data['iid'],
        data[['initial', 'finals', 'tone']]
    ))
    eval_size = int(data.shape[0] * 0.1)
    train_data = dataset.skip(eval_size)
    eval_data = dataset.take(eval_size)

    LinearPredictor(dialects, chars, (initials, finals, tones)) \
        .train(train_data, eval_data)
    MLPPredictor(dialects, chars, (initials, finals, tones)) \
        .train(train_data, eval_data)
    ResidualPredictor(dialects, chars, (initials, finals, tones)) \
        .train(train_data, eval_data)
    AttentionPredictor(dialects, chars, (initials, finals, tones)) \
        .train(train_data, eval_data)


if __name__ == '__main__':
    prefix = sys.argv[1]

    dialect_path = os.path.join(prefix, 'dialect')
    location = pandas.read_csv(
        os.path.join(dialect_path, 'location.csv'),
        index_col=0
    ).sample(100)
    char = pandas.read_csv(os.path.join(prefix, 'words.csv'), index_col=0)
    data = load_data(dialect_path, location.index).sample(frac=0.8)

    benchmark(data)