import numpy as np
import copy

# np.random.seed(5201314)

def get_rotation_array(index):
    if index == 0:
        return np.array([0, 1, 2])
    elif index == 1:
        return np.array([0, 2, 1])
    elif index == 2:
        return np.array([1, 0, 2])
    elif index == 3:
        return np.array([1, 2, 0])
    elif index == 4:
        return np.array([2, 0, 1])
    elif index == 5:
        return np.array([2, 1, 0])
    else:
        return None
def get_rotation_index(rotation):
    if np.array_equal(rotation, np.array([0, 1, 2])):
        return 0
    elif np.array_equal(rotation, np.array([0, 2, 1])):
        return 1
    elif np.array_equal(rotation, np.array([1, 0, 2])):
        return 2
    elif np.array_equal(rotation, np.array([1, 2, 0])):
        return 3
    elif np.array_equal(rotation, np.array([2, 0, 1])):
        return 4
    elif np.array_equal(rotation, np.array([2, 1, 0])):
        return 5
    else:
        return None
    



def bin_packing_problem_generator():
    # container_size=100*100*100
    
    item_original = {
        'shape':    np.array([50,50,50]),  #[length, height, width]
        'position': np.array([0, 0, 0]),   #[x, y, z]
        'rotation': 0,    #0=x, 1=y, 2=z
    }    
    """
    rotation: 
    [0, 1, 2]->0
    [0, 2, 1]->1
    [1, 0, 2]->2
    [1, 2, 0]->3
    [2, 0, 1]->4
    [2, 1, 0]->5   
    """
    
    
    # initialize item list
    items = []
    
    items.append(item_original)
    
    N=np.random.randint(10,51)

    while len(items) < N:
        # pop an item from the list
        item=items.pop(0)
        # choose an axis of the item randomly
        axis=np.random.randint(0,3)


        # Choose a position randomly on the axis by the distance to the center of edge
        position=np.random.randint(-item['shape'][get_rotation_array(item['rotation'])[axis]],item['shape'][get_rotation_array(item['rotation'])[axis]])
        # Split the item into two items.
        items1=copy.deepcopy(item)
        items2=copy.deepcopy(item)

  
        items1['shape'][axis]=(items1['shape'][axis]+position)/2.0
        items2['shape'][axis]=(items2['shape'][axis]-position)/2.0

        items1['position'][axis]=items1['position'][axis]+position-items1['shape'][axis]
        items2['position'][axis]=items2['position'][axis]+position+items2['shape'][axis]
                
        items.append(items1)
        items.append(items2)
    
    for item in items:
        shuffle_ix = np.random.permutation(np.arange(3))
        item['rotation']= get_rotation_index(shuffle_ix)
        item['shape'] = item['shape'][shuffle_ix]*2

    return items


if __name__ == "__main__":
    items_list = bin_packing_problem_generator()
    for item in items_list:
        print(item)