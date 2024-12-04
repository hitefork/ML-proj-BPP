"""
Utilities for the Bin Packing Problem
"""
import random as rd
from copy import deepcopy
from typing import List
import copy
import numpy as np
import math
from nptyping import NDArray, Int, Shape
from operator import *
def get_rotation_array(index):
    if index == 0:
        return [0, 1, 2]
    elif index == 1:
        return [0, 2, 1]
    elif index == 2:
        return [1, 0, 2]
    elif index == 3:
        return [1, 2, 0]
    elif index == 4:
        return [2, 0, 1]
    elif index == 5:
        return [2, 1, 0]
    
    
    
    
    
def get_rotation_index(rotation):
    if eq(rotation,[0, 1, 2]):
        return 0
    elif eq(rotation,[0, 2, 1]):
        return 1
    elif eq(rotation,[1, 0, 2]):
        return 2
    elif eq(rotation,[1, 2, 0]):
        return 3
    elif eq(rotation,[2, 0, 1]):
        return 4
    elif eq(rotation,[2, 1, 0]):
        return 5


def boxes_generator(
    bin_size: List[int]
):
    """Generates instances of the 2D and 3D bin packing problems

    Parameters
    ----------
    bin_size: List[int], optional (default = [10,10,10])
        List of length 2 or 3 with the dimensions of the container (default = (10,10,10))

    Returns
    -------
    List[np.array(np.int32)]
    A list of length num_items with the dimensions of the randomly generated boxes.
    
    List[dict]
    A list of items.
    """

    item_original = {
        'shape':    bin_size,  #[length, height, width]
        'position': [0, 0, 0],   #[x, y, z]
        'rotation':  0,    
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
    
    items_train=[]
    items_label=[]
    
    # initialize item list
    items = []
    
    items.append(item_original)
    
    N=np.random.randint(10,51)
    z=np.zeros((N))
    while len(items) < N:
        # pop an item from the list
        item=items.pop(0)
        # choose an axis of the item randomly
        axis=np.random.randint(0,3)

        if item['shape'][axis] <=1:
            items.append(item)
            continue
        # Choose a position randomly on the axis by the distance to the center of edge
        if item['shape'][axis]<=3:
            position=1
        else:
            position=np.random.randint(1,item['shape'][axis]/2)

        # Split the item into two items.
        items1=copy.deepcopy(item)
        items2=copy.deepcopy(item)

  
        items1['shape'][axis]=position
        items2['shape'][axis]=items2['shape'][axis]-position


        items2['position'][axis]=items2['position'][axis]+position

        items.append(items1)
        items.append(items2)
    
    
    
    for i,item in enumerate(items):
        shuffle_ix = np.random.permutation(np.arange(3)).tolist()
        item['rotation']= get_rotation_index(shuffle_ix)
        item['shape'] = np.array(item['shape'])[shuffle_ix].tolist()
        z[i]=item['position'][2]
    
    z_index=z.argsort().tolist()
    
    for index in z_index:
        items_label.append(items[index])
        items_train.append(items[index]['shape'])
    



    return items_train,items_label


def generate_vertices(
    cuboid_len_edges, cuboid_position
):
    """Generates the vertices of a box or container in the correct format to be plotted

    Parameters
    ----------
    cuboid_position: np.array(np.int32)
          List of length 3 with the coordinates of the back-bottom-left vertex of the box or container
    cuboid_len_edges: np.array(np.int32)
        List of length 3 with the dimensions of the box or container

    Returns
    -------
    np.nd.array(np.int32)
    An array of shape (8,3) with the coordinates of the vertices of the box or container
    """
    # Generate the list of vertices by adding the lengths of the edges to the coordinates
    v0 = cuboid_position

    v1 = v0 + [cuboid_len_edges[0], 0, 0]
    v2 = v0 + [0, cuboid_len_edges[1], 0]
    v3 = v0 + [cuboid_len_edges[0], cuboid_len_edges[1], 0]
    v4 = v0 + [0, 0, cuboid_len_edges[2]]
    v5 = v1 + [0, 0, cuboid_len_edges[2]]
    v6 = v2 + [0, 0, cuboid_len_edges[2]]
    v7 = v3 + [0, 0, cuboid_len_edges[2]]
    vertices = np.vstack((v0, v1, v2, v3, v4, v5, v6, v7))
    return vertices


def interval_intersection(a: List[int], b: List[int]) -> bool:
    """Checks if two open intervals with integer endpoints have a nonempty intersection.

    Parameters
    ----------
    a: List[int]
        List of length 2 with the start and end of the first interval
    b: List[int]
        List of length 2 with the start and end of the second interval

    Returns
    -------
    bool
    True if the intervals intersect, False otherwise
    """
    assert a[1] > a[0], "a[1] must be greater than a[0]"
    assert b[1] > b[0], "b[1] must be greater than b[0]"
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def cuboids_intersection(cuboid_a: List[int], cuboid_b: List[int]) -> bool:
    """Checks if two cuboids have an intersection.

    Parameters
    ----------
    cuboid_a: List[int]
        List of length 6 [x_min_a, y_mina, z_min_a, x_max_a, y_max_a, z_max_a]
        with the start and end coordinates of the first cuboid in each axis

    cuboid_b: List[int]
        List of length 6 [x_min_b, y_min_b, z_min_b, x_max_b, y_max_b, z_max_b]
        with the start and end coordinates of the second cuboid in each axis

    Returns
    -------
    bool
    True if the cuboids intersect, False otherwise
    """
    assert len(cuboid_a) == 6, "cuboid_a must be a list of length 6"
    assert len(cuboid_b) == 6, "cuboid_b must be a list of length 6"

    # Check the coordinates of the back-bottom-left vertex of the first cuboid
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_a[:3])
    ), "cuboid_a must have nonnegative coordinates"
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_b[:3])
    ), "cuboid_b must have non-negative coordinates"

    assert np.all(
        np.less(cuboid_a[:3], cuboid_a[3:])
    ), "cuboid_a must have non-zero volume"

    assert np.all(
        np.less(cuboid_b[:3], cuboid_b[3:])
    ), "cuboid_b must have non-zero volume"

    inter = [
        interval_intersection([cuboid_a[0], cuboid_a[3]], [cuboid_b[0], cuboid_b[3]]),
        interval_intersection([cuboid_a[1], cuboid_a[4]], [cuboid_b[1], cuboid_b[4]]),
        interval_intersection([cuboid_a[2], cuboid_a[5]], [cuboid_b[2], cuboid_b[5]]),
    ]

    return np.all(inter)


def cuboid_fits(cuboid_a: List[int], cuboid_b: List[int]) -> bool:
    """Checks if cuboid_b fits into cuboid_a.
    Parameters
    ----------
    cuboid_a: List[int]
        List of length 6 [x_min_a, y_mina, z_min_a, x_max_a, y_max_a, z_max_a]
        with the start and end coordinates of the first cuboid in each axis
    cuboid_b: List[int]
        List of length 6 [x_min_b, y_min_b, z_min_b, x_max_b, y_max_b, z_max_b]
        with the start and end coordinates of the second cuboid in each axis
    Returns
    -------
    bool
    True if the cuboid_b fits into cuboid_a, False otherwise
    """
    assert len(cuboid_a) == 6, "cuboid_a must be a list of length 3"
    assert len(cuboid_b) == 6, "cuboid_b must be a list of length 3"

    assert len(cuboid_a) == 6, "cuboid_a must be a list of length 6"
    assert len(cuboid_b) == 6, "cuboid_b must be a list of length 6"

    # Check the coordinates of the back-bottom-left vertex of the first cuboid
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_a[:3])
    ), "cuboid_a must have non-negative coordinates"
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_b[:3])
    ), "cuboid_b must have non-negative coordinates"

    assert np.all(
        np.less(cuboid_a[:3], cuboid_a[3:])
    ), "cuboid_a must have non-zero volume"

    assert np.all(
        np.less(cuboid_b[:3], cuboid_b[3:])
    ), "cuboid_b must have non-zero volume"

    # Check if the cuboid b fits into the cuboid a
    return np.all(np.less_equal(cuboid_a[:3], cuboid_b[:3])) and np.all(
        np.less_equal(cuboid_b[3:], cuboid_a[3:])
    )


if __name__ == "__main__":
    items,items_list = boxes_generator([100,100,100])
    print(items)
    for item in items_list:
        print(item)
