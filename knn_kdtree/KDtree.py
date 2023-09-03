import numpy as np
import collections
import operator
from collections import Counter

#Constructor de KD tree
BT = collections.namedtuple("BT", ["value", "left", "right"])

def kdTree(arr):
  k = len(arr[1])

  def build(*, arr, depth):
    if len(arr) == 0:
      return None
    
    arr.sort(key=operator.itemgetter(depth % k))

    middle = len(arr) // 2

    return BT(
      value = arr[middle],
      left = build(arr = arr[:middle], depth = depth + 1),
      right = build(arr= arr[middle + 1:], depth = depth + 1)
    )
  return build(arr = list(arr), depth = 0)

# Square Euclidean distance
def SED(X, Y):
    return sum((i-j)**2 for i, j in zip(X, Y))

# Puntos rama
def preOrden(tree, point, depth, points):
    k = len(point)
    cd = depth % k
    if not tree:
        return
    
    if arePointsSame(tree.value, point):
        points.append(tree.value)
        return points
 
    if point[cd] < tree.value[cd]:
      preOrden(tree.left, point, depth + 1, points)
      points.append(tree.value)
    else:
      preOrden(tree.right, point, depth + 1, points)
      points.append(tree.value)
    
    if point[cd] - tree.value[cd] < SED(point, tree.value):
      preOrden(tree.right, point, depth + 1, points)
      points.append(tree.value)
    else:
      preOrden(tree.left, point, depth + 1, points)
      points.append(tree.value)
    return points


"""
def preOrden(tree, point, depth, points):
    if not tree:
        return
    else:
      preOrden(tree.left, point, depth + 1, points)
      points.append(tree.value)   

    preOrden(tree.right, point, depth + 1, points)
    points.append(tree.value)
    return points
"""

def arePointsSame(point1, point2):
    k = len(point1)
    for i in range(k):
        if point1[i] != point2[i]:
            return False

    return True

# Set de lista de puntos
def pointList(arr):
  arr2 = arr[0:1]
  for i in range(len(arr)):
    if arr[i] not in arr2:
        arr2.append(arr[i])
  return arr2

NNRecord = collections.namedtuple("NNRecord", ["distance", "point"])

class KNNpoints:
  def __init__(self, k, newPoint, tree):
    self.k = k
    self.newPoint = newPoint
    self.tree = tree
    self.point = None
    self.arr1 = []
    self.points = pointList(preOrden(self.tree, self.newPoint, 0, self.arr1))

  def predict(self):
    result = None
    NNresult = []


    for point in self.points:
      distance = SED(point, self.newPoint)
      result = NNRecord(point=point, distance=distance)
      NNresult.append(result)

    NNresult.sort(key=operator.itemgetter(0))
    NNresult = NNresult[:self.k]

    print(NNresult)

    classResult = []

    for i in range(self.k):
        classResult.append(NNresult[i].point)
    return classResult

#Visualizador de árbol
def viewTree(tree, cont = int):
  if tree is None:
    return
  else:
    viewTree(tree.right, cont + 1)
    for i in range(cont):
      print(end = "\t \t")
    print(tree.value, "\n")
    viewTree(tree.left, cont + 1)
  
if __name__ == '__main__':
  
  cont = 0
  tree = None
  reference_points = [[100, 100], [40, 70], [175, 100], [90, 40], [70, 130], [150, 30], [140, 110]]
 
  tree = kdTree(reference_points)
  point = [140, 90]
  knn = KNNpoints(5, point, tree)

  viewTree(tree, cont)

  print("\n")
  print("\n")

  result = knn.predict()

  print(result)
  print("\n")

  print(preOrden(tree,point, 0, points=[]))
  print("\n")
