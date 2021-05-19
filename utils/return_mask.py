import numpy as np

def return_mask_of_data(data):
    mask = 1. - np.isnan(data)
    return mask

def concat_embedding_mask_for_columns(array_location, label_reverse, mask, column_location):
        
        for index, i in enumerate(label_reverse):
            if label_reverse[index][0] == 'con':
                if index == 0:
                    mask_embed = mask[:, 0:column_location[index]]
                else:
                    mask_embed = np.concatenate((mask_embed, mask[:, column_location[index-1]:column_location[index]]), axis = 1)

            else:
                if index == 0:
                    mask_embed = mask[:, 0:1]
                    for j in range(array_location[index] - 1):
                        mask_embed = np.concatenate((mask_embed, mask[:, 0:1]), axis = 1)
                else:
                    for j in range(array_location[index - 1], array_location[index]):
                        mask_embed = np.concatenate((mask_embed, mask[:, column_location[index-1]:column_location[index-1] + 1]), axis = 1)
                        
        return mask_embed

         
def return_array_size_and_location_for_columns_embedding(label_reverse, mask, column_location):

    array_size = 0
    array_location = []
    
    length_embedding_min = 5
    length_embedding_max = 10

    for index, i in enumerate(label_reverse):
        if i[0] == 'con':
            array_size = array_size + 1
            array_location.append(array_size)
        elif i[0] == 'cat':
            length_pre = len(i[1][0])
            if length_pre <= length_embedding_min:
                length = length_embedding_min
            elif length_pre <= length_embedding_max and length_pre > length_embedding_min:
                length = length_pre
            else:
                length = length_embedding_max
            array_size = array_size + length
            array_location.append(array_size)

    embedding_mask = concat_embedding_mask_for_columns(array_location = array_location, label_reverse = label_reverse, mask = mask, column_location = column_location)

    return array_size, array_location, embedding_mask



    
