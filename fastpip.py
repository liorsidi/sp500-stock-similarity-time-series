# -*- coding: utf-8 -*-
import logging as default_logging
import sys
import math

__all__ = ['pip']

logging = default_logging.getLogger('simentarweb:utils:charts')

INFINITY = float('inf')


class PipItem(object):

    def __init__(self, index, parent, **kwargs):
        self.index = index
        self.parent = parent
        self.cache = kwargs.get('cache', INFINITY)
        self.value = kwargs.get('value', None)
        self.order = kwargs.get('order', None)
        self.left = kwargs.get('left', None)
        self.right = kwargs.get('right', None)

    def update_cache(self):
        if None in (self.left, self.right):
            self.cache = INFINITY
        else:
            self.cache = self.parent.distance(self.left.value, self.value, self.right.value)
        self.parent.notify_change(self.index)

    def put_after(self, tail):
        if tail is not None:
            tail.right = self
            tail.update_cache()
        self.left = tail
        self.update_cache()
        return self

    def recycle(self):
        if self.left is None:
            self.parent.head = self.right
        else:
            self.left.right = self.right
            self.left.update_cache()

        if self.right is None:
            self.parent.tail = self.left
        else:
            self.right.left = self.left
            self.right.update_cache()

        return self.clear()

    def clear(self):
        self.order = 0
        self.left = None
        self.right = None
        self.cache = INFINITY

        ret = self.value
        self.value = None
        return ret


class PipHeap(object):

    def __init__(self, distance, **kwargs):
        self.distance = distance
        self.heap = self.create_heap(512)
        self.head = None
        self.tail = None
        self.size = 0
        self.global_order = 0

    def create_heap(self, size):
        return [PipItem(i, self) for i in range(size)]

    def ensure_heap(self, size):
        new_elements = [PipItem(i, self) for i in range(len(self.heap), size+1)]
        self.heap.extend(new_elements)

    def acquire_item(self, value):
        self.ensure_heap(self.size)
        item = self.heap[self.size]
        item.value = value

        self.size += 1
        self.global_order += 1
        item.order = self.global_order
        return item

    def add(self, value):
        self.tail = self.acquire_item(value).put_after(self.tail)
        if self.head is None:
            self.head = self.tail

    @property
    def min_value(self):
        return self.heap[0].cache

    def remove_min(self):
        return self.remove_at(0)

    def remove_at(self, index):
        self.size -= 1
        self.swap(index, self.size)
        self.bubble_down(index)
        return self.heap[self.size].recycle()

    def notify_change(self, index):
        return self.bubble_down(self.bubble_up(index))

    def bubble_up(self, n):
        while (n != 0) and self.less(n, (n-1)/2):
            n = self.swap(n, (n-1)/2)
        return n

    def bubble_down(self, n):
        get_k = lambda n: self.min(n, n*2+1, n*2+2)

        k = get_k(n)
        while (k != n) and (k < self.size):
            n = self.swap(n, k)
            k = get_k(n)
        return n

    def min(self, i, j, k=None):
        if k is not None:
            result = self.min(i, self.min(j, k))
        else:
            result = i if self.less(i, j) else j

        return result

    def less(self, i, j):
        def i_smaller_than_j(heap, i, j):
            i, j = int(i), int(j)
            if heap[i].cache != heap[j].cache:
                result = heap[i].cache < heap[j].cache
            else:
                result = heap[i].order < heap[j].order
            return result

        heap = self.heap
        return ((i < self.size) and (j >= self.size or i_smaller_than_j(heap, i, j)))

    def swap(self, i, j):
        i, j = int(i), int(j)
        self.heap[i].index, self.heap[j].index = j, i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        return j

    def __iter__(self):
        current = self.head
        while current is not None:
            yield current.value
            current = current.right


def vertical_distance(left, current, right):
    EPSILON = 1e-06
    a_x, a_y = left
    b_x, b_y = current
    c_x, c_y = right

    if (abs(a_x - b_x) < EPSILON) or (abs(b_x - c_x) < EPSILON):
        result = 0
    elif (c_x - a_x) == 0:
        # Otherwise we could have a ZeroDivisionError
        result = INFINITY
    else:
        result = abs(((a_y + (c_y - a_y) * (b_x - a_x) / (c_x - a_x) - b_y)) * (c_x - a_x))

    return result


def dist(x1, x2, y1, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def euclidean_distance(left, current, right):
    left_current = dist(left[0], current[0], left[1], current[1])
    rightcurrent = dist(right[0], current[0], right[1], current[1])
    return (left_current + rightcurrent) * (right[0] - left[0])


def pip(data, k, fast=True, stream_mode=True, distance='vertical'):
    distance_functions = {
        'vertical': vertical_distance,
        'euclidean': euclidean_distance,
    }
    distance_function = distance_functions[distance]

    if fast:
        result = fastpip(data, k, stream_mode=stream_mode, distance_function=distance_function)
    else:
        result = simplepip(data, k, distance_function=distance_function)

    return result


def fastpip(data, k, stream_mode=True, distance_function=vertical_distance):

    if len(data) >= k:
        heap = PipHeap(distance_function)

        for element in data:
            heap.add(element)

            if stream_mode and (heap.size > k):
                heap.remove_min()

        if not stream_mode:
            while heap.size > k:
                heap.remove_min()

        ret = list(heap)
    else:
        ret = data

    logging.debug("pip: started with {} points, returned {} points".format(len(data), len(ret)))
    return ret


def simplepip(data, k, distance_function=vertical_distance):
    ret = []

    for (idx, value) in enumerate(data):
        ret.append(value)
        if len(ret) <= k:
            continue

        miniv = sys.maxsize
        minij = 0

        for j in range(1, len(ret) - 1):
            d = distance_function(ret[j - 1], ret[j], ret[j + 1])
            if d < miniv:
                miniv = d
                minij = j

        del ret[minij]

    return ret