import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Dropout, Activation, LayerNormalization, Embedding, Reshape
from tensorflow.python.keras.models import Sequential, Model

def return_layer(layer_input, output_size, norm = False, dropout = False, activation = 'relu'): 
    output = Dense(output_size, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(layer_input)
    if norm:
        output = LayerNormalization()(output)
    else:
        output = output

    if dropout:
        output = Dropout(rate = dropout)(output)
    else:
        output = output

    if activation:
        output = Activation(activation = activation)(output)
    else:
        output = output

    return output

def return_embedding_model_for_columns(column_label_reverse):
        
    # The embedding size of each discrete variable is length.
    # Return the embedding model with 1 input and embedding size output.

    length_model = len(column_label_reverse[1][0])
    length_pre = len(column_label_reverse[1][0])
    
    length_embedding_min = 5
    length_embedding_max = 10

    if length_pre <= length_embedding_min:
        length = length_embedding_min
    elif length_pre <= length_embedding_max and length_pre > length_embedding_min:
        length = length_pre
    else:
        length = length_embedding_max

    model = Sequential()
    model.add(Input(shape = (1, )))
    model.add(Embedding(input_dim = length_model, output_dim = length, input_length = 1))
    model.add(Reshape((length, )))

    return model
