import tensorflow as tf
import my_utils
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.layers import Embedding
from keras.layers import Input, Dense, merge
import keras.backend as K
import preprocess_SemCor as sm
import  numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras import optimizers
from generator import *
import keras.backend as k
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import my_utils as ut
import os
import sys
from keras.utils import plot_model
from keras import metrics
import tensorflow as tf


EMBEDDING_DIM=100

def custom_categorical_crossentropy(y_true, y_pred):
    print(y_true.shape, " èllaipsilon",file=sys.stderr)
    #return y_true
    i = k.argmax(y_true)
    vect_value = k.max(y_true)
    return K.categorical_crossentropy(y_true, y_pred).__mul__(vect_value)




print("build model")

#--- build the model ---

def create_simple_keras_model(emb_input_dim, num_outputs_syn, emb_output_dim=32, LSTM_hl = 64, lr = 0.005, verbose = False
                              , n_stacked=1, LSTM_dropout= 0.0, weights = None, attention = False):

    sentence_input = Input(shape=(None,), dtype=tf.int64)

    # sentence_embedding = ELMoEmbedding(idx2word=x_reverse_dict, output_mode="elmo", trainable=False)(sentence_input) # These two are interchangeable
    if weights is not None:
        sentence_embedding = Embedding(emb_input_dim + 1,
                                    EMBEDDING_DIM,
                                    weights=[weights],
                                    trainable=False)(sentence_input)
    else:
        sentence_embedding = Embedding(emb_input_dim, emb_output_dim, mask_zero=True)(sentence_input)

    stacked_lstm = LSTM(LSTM_hl, return_sequences=True, go_backwards=True,
                        recurrent_dropout=LSTM_dropout)(sentence_embedding)

    stacked_lstm, state_h, state_c = LSTM(LSTM_hl, return_sequences=True, go_backwards=False,
                      recurrent_dropout=LSTM_dropout, return_state=True)(stacked_lstm)



    for i in range(n_stacked-1):
        stacked_lstm = LSTM(LSTM_hl, return_sequences=True, go_backwards=True,
                            recurrent_dropout=LSTM_dropout)(stacked_lstm)
        stacked_lstm = LSTM(LSTM_hl, return_sequences=True, go_backwards=False,
                            recurrent_dropout=LSTM_dropout)(stacked_lstm)
    if attention:
        #--------- ATTENTION LAYER XD -----------------
        attention = Dense(1, activation='tanh')(state_h)
        #attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(LSTM_hl)(attention)
        attention = Permute([2, 1])(attention)
        sent_representation = keras.layers.multiply([stacked_lstm, attention])
        out_syn = TimeDistributed(Dense(num_outputs_syn, activation="softmax"))(sent_representation)
    else:
        out_syn = TimeDistributed(Dense(num_outputs_syn, activation="softmax"))(stacked_lstm)

    optimizer = optimizers.Adadelta(lr=lr)
    model = Model(inputs=sentence_input, outputs=out_syn)
    model.compile(loss=custom_categorical_crossentropy, optimizer=optimizer, metrics=[''])
    if verbose:
        model.summary()
    return model


def create_simple_keras_model_nltk(emb_input_dim, num_outputs_syn, emb_output_dim=32, LSTM_hl=64, lr=0.005, verbose=False
                              , n_stacked=1, LSTM_dropout=0.0, weights=None):
    word_input = Input(shape=(None,), dtype=tf.int64)
    print("lool ",sentence_input.shape)
    # sentence_embedding = ELMoEmbedding(idx2word=x_reverse_dict, output_mode="elmo", trainable=False)(sentence_input) # These two are interchangeable
    
    if weights is not None:
        word_embedding = Embedding(emb_input_dim_w + 1,
                                       EMBEDDING_DIM,
                                       weights=[weights],
                                       trainable=False)(word_input)
    else:
        word_embedding = Embedding(emb_input_dim_w, emb_output_dim, mask_zero=True)(word_input)

    
    out_syn = TimeDistributed(Dense(num_outputs_syn, activation="softmax"))(stacked_lstm)

    #out_syn = TimeDistributed(Dense(num_outputs_syn, activation="softmax", input_dim=2))(stacked_lstm)

    optimizer = optimizers.Adam(lr=lr)
    model = Model(inputs=sentence_input, outputs=out_syn)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    if verbose:
        model.summary()
    return model


def create_simple_keras_model_final(emb_input_dim_w, emb_input_dim_pos,  num_outputs_syn, emb_output_dim=32, LSTM_hl=64, lr=0.005, verbose=False
                                       , n_stacked=1, LSTM_dropout=0.0, weights=None):
    word_input = Input(shape=(None,), dtype=tf.int64)
    print("lool ",word_input.shape)
    # sentence_embedding = ELMoEmbedding(idx2word=x_reverse_dict, output_mode="elmo", trainable=False)(sentence_input) # These two are interchangeable
    if weights is not None:
        word_embedding = Embedding(emb_input_dim_w + 1,
                                       EMBEDDING_DIM,
                                       weights=[weights],
                                       trainable=False)(word_input)
    else:
        word_embedding = Embedding(emb_input_dim_w, emb_output_dim, mask_zero=True)(word_input)

    # combine the output of the two branches



    #stacked_lstm = LSTM(LSTM_hl, return_sequences=True, go_backwards=True,
     #                   recurrent_dropout=LSTM_dropout)(combined)
    stacked_lstm = Bidirectional(LSTM(LSTM_hl, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(word_embedding)
    #stacked_lstm = LSTM(LSTM_hl, return_sequences=True, go_backwards=False,
     #                   recurrent_dropout=LSTM_dropout)(stacked_lstm)

    #stacked_lstm, state_h, state_c = LSTM(LSTM_hl, return_sequences=True, go_backwards=False,
    #                                      recurrent_dropout=LSTM_dropout, return_state=True)(stacked_lstm)

    for i in range(n_stacked - 1):
        stacked_lstm = Bidirectional(LSTM(LSTM_hl, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(stacked_lstm)

    out_syn = TimeDistributed(Dense(num_outputs_syn, activation="softmax"))(stacked_lstm)

    #out_syn = TimeDistributed(Dense(num_outputs_syn, activation="softmax", input_dim=2))(stacked_lstm)

    optimizer = optimizers.Adam(lr=lr)
    model = Model(inputs=word_input, outputs=out_syn)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    if verbose:
        model.summary()
    return model



def create_simple_keras_model_with_POS(emb_input_dim_w, emb_input_dim_pos,  num_outputs_syn, emb_output_dim=32, LSTM_hl=64, lr=0.005, verbose=False
                                       , n_stacked=1, LSTM_dropout=0.0, weights=None):
    word_input = Input(shape=(None,), dtype=tf.int64)
    POS_input = Input(shape=(None,), dtype=tf.int64)

    print("lool ",word_input.shape)
    # sentence_embedding = ELMoEmbedding(idx2word=x_reverse_dict, output_mode="elmo", trainable=False)(sentence_input) # These two are interchangeable
    if weights is not None:
        word_embedding = Embedding(emb_input_dim_w + 1,
                                       EMBEDDING_DIM,
                                       weights=[weights],
                                       trainable=False)(word_input)
    else:
        word_embedding = Embedding(emb_input_dim_w, emb_output_dim, mask_zero=True)(word_input)

    POS_embedding = Embedding(emb_input_dim_pos, emb_output_dim, mask_zero=True)(POS_input)

    # combine the output of the two branches

    x = Model(inputs=word_input, outputs=word_embedding)
    y = Model(inputs=POS_input, outputs=POS_embedding)
    combined = concatenate([x.output, y.output])


    #stacked_lstm = LSTM(LSTM_hl, return_sequences=True, go_backwards=True,
     #                   recurrent_dropout=LSTM_dropout)(combined)
    stacked_lstm = Bidirectional(LSTM(LSTM_hl, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(combined)
    #stacked_lstm = LSTM(LSTM_hl, return_sequences=True, go_backwards=False,
     #                   recurrent_dropout=LSTM_dropout)(stacked_lstm)

    #stacked_lstm, state_h, state_c = LSTM(LSTM_hl, return_sequences=True, go_backwards=False,
    #                                      recurrent_dropout=LSTM_dropout, return_state=True)(stacked_lstm)

    for i in range(n_stacked - 1):
        stacked_lstm = Bidirectional(LSTM(LSTM_hl, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(stacked_lstm)

    out_syn = TimeDistributed(Dense(num_outputs_syn, activation="softmax"))(stacked_lstm)

    #out_syn = TimeDistributed(Dense(num_outputs_syn, activation="softmax", input_dim=2))(stacked_lstm)

    optimizer = optimizers.Adam(lr=lr)
    model = Model(inputs=[word_input,POS_input], outputs=out_syn)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    if verbose:
        model.summary()
    return model




def create_MTL_keras_model(emb_input_dim, num_outputs_syn, num_output_lex, num_output_wnd, emb_output_dim=32,
                           LSTM_hl = 64, lr = 0.005, verbose = False, n_stacked=1, LSTM_dropout= 0.0):
    # sentence_input = Input(shape=(x_data_train.shape[1],), dtype=tf.int64)
    sentence_input = Input(shape=(None,), dtype=tf.int64)

    # sentence_embedding = ELMoEmbedding(idx2word=x_reverse_dict, output_mode="elmo", trainable=False)(sentence_input) # These two are interchangeable
    sentence_embedding = Embedding(emb_input_dim, emb_output_dim, mask_zero=True)(sentence_input)

    stacked_lstm = LSTM(LSTM_hl, return_sequences=True, go_backwards=True,
                        recurrent_dropout=LSTM_dropout)(sentence_embedding)

    stacked_lstm = LSTM(LSTM_hl, return_sequences=True, go_backwards=False,
                      recurrent_dropout=LSTM_dropout)(stacked_lstm)

    for i in range(n_stacked-1):
        stacked_lstm = LSTM(LSTM_hl, return_sequences=True, go_backwards=True,
                            recurrent_dropout=LSTM_dropout)(stacked_lstm)
        #layer = Lambda(lambda x: K.reverse(x, axes=1))
        #stacked_lstm = layer(stacked_lstm)


    out_syn = TimeDistributed(Dense(num_outputs_syn, activation="softmax"))(stacked_lstm)
    out_lex = TimeDistributed(Dense(num_output_lex, activation="softmax"))(stacked_lstm)
    out_wnd = TimeDistributed(Dense(num_output_wnd, activation="softmax"))(stacked_lstm)



    optimizer = optimizers.Adam(lr=lr)
    model = Model(inputs=sentence_input, outputs=[out_syn,out_lex,out_wnd])
    model.compile(loss=[custom_categorical_crossentropy,custom_categorical_crossentropy,
                        custom_categorical_crossentropy], optimizer=optimizer, metrics=['categorical_accuracy'])
    if verbose:
        model.summary()
    return model





def train_MTL_model(x_path, y_path_syn, y_path_lex, y_path_wnd, x_val_path, y_val_path_syn,
                    y_val_path_lex, y_val_path_wnd, gold_path, output_path, scorer_path ):
    '''
    x, x_fin, _, _ = sm.preprocess_SemCor_final(x_path)
    y_syn = sm.obtain_y_value_final(x_fin, y_path_syn)
    print(y_syn[0])
    y_lex = sm.obtain_y_value_nltk(x_fin, y_path_lex)
    y_wnd = sm.obtain_y_value_nltk(x_fin, y_path_wnd)

    x_val, x_val_fin,_, _ =  sm.preprocess_SemCor_final(x_val_path)
    y_val_syn = sm.obtain_y_value_final(x_val_fin, y_val_path_syn)
    y_val_lex = sm.obtain_y_value_nltk(x_val_fin, y_val_path_lex)
    y_val_wnd = sm.obtain_y_value_nltk(x_val_fin, y_val_path_wnd)

    _, y_syn_dict, y_syn_reverse_dict = sm.build_y_dataset(y_syn + y_val_syn)
    _, y_lex_dict, y_lex_reverse_dict = sm.build_y_dataset(y_lex + y_val_lex)
    _, y_wnd_dict, y_wnd_reverse_dict = sm.build_y_dataset(y_wnd + y_val_wnd)

    x_data_train, x_dict, x_reverse_dict = sm.build_x_dataset(x, x_fin)
    print("len dict x ",len(x_dict))

    y_syn_data, _, _ = sm.build_y_dataset(y_syn, dictionary=y_syn_dict)
    y_lex_data, _, _ = sm.build_y_dataset(y_lex, dictionary=y_lex_dict)
    y_wnd_data, _, _ = sm.build_y_dataset(y_wnd, dictionary=y_wnd_dict)

    x_data_val, _, _ = sm.build_x_dataset(x_val, dictionary=x_dict)

    y_syn_data_val, _, _ = sm.build_y_dataset(y_val_syn, dictionary=y_syn_dict)
    y_lex_data_val, _, _ = sm.build_y_dataset(y_val_lex, dictionary=y_lex_dict)
    y_wnd_data_val, _, _ = sm.build_y_dataset(y_val_wnd, dictionary=y_wnd_dict)
    #for i in range(len(x_data_train)):
     #   print (x_data_train[i])
      #  print (y_syn_data[i])
    '''
    '''
    training_generator = BaseGenerator_MTL(x_data_train, y_syn_data, y_lex_data, y_wnd_data, len(y_syn_dict.keys()),
                                           len(y_lex_dict.keys()), len(y_wnd_dict), ignore_value_syn=y_syn_dict["PASS"],
                                           ignore_value_lex=y_lex_dict["PASS"], ignore_value_wnd=y_wnd_dict["PASS"],
                                      batch_size=16)

    validation_generator = BaseGenerator_MTL(x_data_val, y_syn_data_val, y_lex_data_val, y_wnd_data_val,
                                             len(y_syn_dict.keys()),
                                             len(y_lex_dict.keys()), len(y_wnd_dict),
                                             ignore_value_syn=y_syn_dict["PASS"],
                                             ignore_value_lex=y_lex_dict["PASS"], ignore_value_wnd=y_wnd_dict["PASS"],
                                             batch_size=16)
    '''

    '''
    training_generator = BaseGenerator(x_data_train, y_syn_data,  len(y_syn_dict.keys()),
                                       ignore_value=y_syn_dict["PASS"],
                                           batch_size=16)


    validation_generator = BaseGenerator(x_data_val, y_syn_data_val,  len(y_syn_dict.keys()),
                                       ignore_value=y_syn_dict["PASS"],
                                            batch_size=16)
    '''
    '''
    training_generator = BaseGenerator_nltk(x_data_train, y_syn_data,  len(y_syn_dict.keys()),
                                           batch_size=32)


    validation_generator = BaseGenerator_nltk(x_data_val, y_syn_data_val,  len(y_syn_dict.keys()),
                                            batch_size=32)
    '''




    
    x, x_fin, sentence_pos, _ = sm.preprocess_SemCor_final(x_path)
    y_syn = sm.obtain_y_value_final(x_fin, y_path_syn)
    print(y_syn[0])


    x_val, x_val_fin,sentence_pos_val, _ =  sm.preprocess_SemCor_final(x_val_path)
    y_val_syn = sm.obtain_y_value_final(x_val_fin, y_val_path_syn)


    _, y_syn_dict, y_syn_reverse_dict = sm.build_y_dataset(y_syn + y_val_syn)

    x_data_POS, x_dict_POS, x_reverse_dict_POS = sm.build_x_dataset(sentence_pos, x_fin)
    x_data_train, x_dict, x_reverse_dict = sm.build_x_dataset(x, x_fin)
    print("len dict x ",len(x_dict))

    y_syn_data, _, _ = sm.build_y_dataset(y_syn, dictionary=y_syn_dict)
    #y_lex_data, _, _ = sm.build_y_dataset(y_lex, dictionary=y_lex_dict)
    #y_wnd_data, _, _ = sm.build_y_dataset(y_wnd, dictionary=y_wnd_dict)
    x_data_POS_val, _, _= sm.build_x_dataset(sentence_pos_val, x_fin)

    x_data_val, _, _ = sm.build_x_dataset(x_val, dictionary=x_dict)

    y_syn_data_val, _, _ = sm.build_y_dataset(y_val_syn, dictionary=y_syn_dict)

   # training_generator = BaseGenerator_with_POS(x_data_train,x_data_POS, y_syn_data,  len(y_syn_dict.keys()),
    #                                       batch_size=32)


    #validation_generator = BaseGenerator_with_POS(x_data_val,x_data_POS_val, y_syn_data_val,  len(y_syn_dict.keys()),
     #                                       batch_size=32)
    training_generator = BaseGenerator_nltk(x_data_train, y_syn_data,  len(y_syn_dict.keys()),
                                           batch_size=32)


    validation_generator = BaseGenerator_nltk(x_data_val, y_syn_data_val,  len(y_syn_dict.keys()),batch_size=32)
     


    #model = create_MTL_keras_model(len(x_reverse_dict.keys()), len(y_syn_dict.keys()), len(y_lex_dict.keys()),
     #                             len(y_wnd_dict.keys()), verbose=True)
    model_path =r"C:\Users\barfo\Desktop\appnmodello.png"
    print("lunghezze ",len(x_dict.keys()),"  ", len(y_syn_dict.keys()))
    #model = create_simple_keras_model_nltk(len(x_reverse_dict.keys()), len(y_syn_dict.keys()), LSTM_hl=128,lr=0.05,n_stacked=1,LSTM_dropout=0.1)
    print("lunghezza y-syn",len(y_syn_dict.keys()))
    model = create_simple_keras_model_final(len(x_reverse_dict.keys()),len(x_reverse_dict_POS.keys()), len(y_syn_dict.keys()), LSTM_hl=512,lr=0.003, emb_output_dim=128,n_stacked=1)

    #plot_model(model, to_file=model_path, show_shapes=True)
    print(type(y_syn_data), "  ", type(y_syn_data[0]))
    callbacks = [EarlyStopping('val_loss', patience=2)]
    callbacks += [
        ModelCheckpoint("weights.{epoch:02d}-{categorical_accuracy:.5f}.hdf5",
                        monitor='categorical_accuracy', verbose=1, save_best_only=False,
                        save_weights_only=True, mode='auto')]
    model.summary()
    model.load_weights(r"/home/Francesco/azure_work_final/code/weights.01-0.75077.hdf5")
    for i in range (20):

        pred = ut.predict_single(x_val_path, model, None, y_syn_dict, x_dict, output_path)
        a = ut.compute_score(gold_path, output_path, scorer_path)
        print(a)
        history = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                                  epochs=1, callbacks=callbacks)

        pred = ut.predict_single(x_val_path, model, None, y_syn_dict, x_dict, output_path)

        a = ut.compute_score(gold_path, output_path, scorer_path)
        print(a)


def create_pretrained_matrix(GLOVE_DIR, x_dict, MAX_NUM_WORDS = 20_000):
    #num_words = min(MAX_NUM_WORDS, len(x_dict.keys())) + 1

    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(x_dict) + 1, EMBEDDING_DIM))
    for word, i in x_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

if __name__ == '__main__':

    x_path = r"C:\Users\barfo\Desktop\appnti nlp3\WSD_Evaluation_Framework\Training_Corpora\SemCor\semcor.data.xml"
    y_path_syn = r"C:\Users\barfo\Desktop\appnti nlp3\y_to_wordent.txt"
    x_val_path = r"C:\Users\barfo\Desktop\appnti nlp3\WSD_Evaluation_Framework\Evaluation_Datasets\semeval2007\semeval2007.data.xml"
    y_val_path_syn = r"C:\Users\barfo\Desktop\appnti nlp3\y_val_2007.txt"

    y_path_lex = r"C:\Users\barfo\Desktop\appnti nlp3\y_to_lex.txt"
    y_path_wnd = r"C:\Users\barfo\Desktop\appnti nlp3\y_to_wnd.txt"

    y_val_path_lex = r"C:\Users\barfo\Desktop\appnti nlp3\y_val_2007_lex.txt"
    y_val_path_wnd = r"C:\Users\barfo\Desktop\appnti nlp3\y_val_2007_wnd.txt"

    output_path = r"C:\Users\barfo\Desktop\appnti nlp3\predictions.txt"
    scorer_path = r"C:\Users\barfo\Desktop\appnti nlp3\WSD_Evaluation_Framework\Evaluation_Datasets"



    train_MTL_model(x_path, y_path_syn, y_path_lex, y_path_wnd, x_val_path, y_val_path_syn, y_val_path_lex,
                    y_val_path_wnd, y_val_path_syn, output_path, scorer_path)


    #model.load_weights(r"C:\Users\barfo\Desktop\public_homework_3\code\keras_elmo_embedding_layer\weights.01-0.01629.hdf5")
    #model.load_weights(r"C:\Users\barfo\Desktop\public_homework_3\code\weights.02-0.62020.hdf5")



    mfs_path = r"C:\Users\barfo\Desktop\appnti nlp3\MFS_vocab.json"

    #ut.create_MFS_dict(x_path,y_path, out_path=mfs_path)

    output_path = r"C:\Users\barfo\Desktop\appnti nlp3\predictions.txt"


    x_val_path = r"C:\Users\barfo\Desktop\appnti nlp3\WSD_Evaluation_Framework\Evaluation_Datasets\semeval2007\semeval2007.data.xml"

    y_val_path_syn= r"C:\Users\barfo\Desktop\appnti nlp3\WSD_Evaluation_Framework\Evaluation_Datasets\semeval2007\semeval2007.gold.key.txt"
    out_y = r"C:\Users\barfo\Desktop\appnti nlp3\y_val_2007.txt"
    #sm.convert_Raganato_into_Wordnet(y_val_path, out_y)

    #pred = ut.predict(x_val_path, model,mfs_path, y_dict, x_dict,output_path)

    scorer_path = r"C:\Users\barfo\Desktop\appnti nlp3\WSD_Evaluation_Framework\Evaluation_Datasets"
    gold_path = r"C:\Users\barfo\Desktop\appnti nlp3\y_val_2007.txt"
    dest_dir =r"C:\Users\barfo\Desktop\appnti nlp3\gridRes"


    #ut.grid_search(x_path, y_path, x_val_path, y_val_path, x_val_path, dest_dir, y_val_path, scorer_path)

    #print(len(pred), "lenpred", len(x_val))
    ut.predict_with_baseline(x_val_path,output_path)
    print(ut.compute_score(out_y,output_path,scorer_path))
