import numpy as np
import copy

# np.random.seed(5201314)

def bin_packing_problem_generator():
    # container_size=100*100*100
    
    item_original = {
        'shape':    np.array([50,50,50]),  #[length, height, width]
        'position': np.array([0, 0, 0]),   #[x, y, z]
        'rotation': np.array([0, 1, 2]),    #0=x, 1=y, 2=z
    }    
    
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
        position=np.random.randint(-item['shape'][item['rotation'][axis]],item['shape'][item['rotation'][axis]])
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
        shuffle_ix = np.random.permutation(np.arange(len(item['rotation'])))
        item['rotation'] = item['rotation'][shuffle_ix]
        item['shape'] = item['shape'][shuffle_ix]

    return items


if __name__ == "__main__":
    items_list = bin_packing_problem_generator()
    for item in items_list:
        print(item)