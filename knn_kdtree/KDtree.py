import numpy as np
import collections
import operator
from collections import Counter

#CREACION DE KD TREE
#**----------------------------------------------------**#
class Node:
    def __init__(self, point, father):
        self.point = point
        self.left = None
        self.right = None
        self.father = father

def newNode(point, father):
    return Node(point, father)

def insert(root, points, depth, father):
    if len(points) == 0:
        return None
    
    k = len(points[0])

    points.sort(key=operator.itemgetter(depth % k))
    middle = len(points) // 2

    if root is None:
        root = newNode(points[middle], father)
    
    if root.left is None:
        root.left = insert(root.left, points[:middle], depth + 1, root)

    if root.right is None:
        root.right = insert(root.right, points[middle + 1:], depth + 1, root)

    return root
#**----------------------------------------------------**#


#BUSQUEDA DE RAMA ELEMENTO
#**----------------------------------------------------**#
def searchRec(root, point, depth, arrPoints):
    k = len(point)
    
    if not root:
        return
    
    cd = depth % k

    if root.point[cd] < point[cd]:
        searchRec(root.right, point, depth + 1, arrPoints)
    else:
        searchRec(root.left, point, depth + 1, arrPoints)

    arrPoints.append(root.point)
    return arrPoints

def search(root, point, arrPoints):
    return searchRec(root, point, 0, arrPoints)
#**----------------------------------------------------**#

#ELIMINAR ELEMENTO DE ARBOL
#**----------------------------------------------------**#
def minleft(root):
    if root is None:
      return None
    if root.left is not None:
      return minleft(root.left)
    else:
      return root

def replaceNd(root, newroot):
    if root.father is not None:
        if root.point == root.father.left.point:
          root.father.left = newroot
        elif root.point == root.father.right.point:
          root.father.right = newroot
    
    if newroot is not None:
       newroot.father = root.father

def dropNode(root):
   root.left = None
   root.right = None
   del root
   
    
def deleteNode(delNode):
    if delNode.left is not None and delNode.right is not None:
        minNode = minleft(delNode.right)
        delNode.point = minNode.point
        deleteNode(minNode)

    elif delNode.left is not None:
        replaceNd(delNode, delNode.left)
        dropNode(delNode)

    elif delNode.right is not None:
        replaceNd(delNode, delNode.right)
        dropNode(delNode)
    else:
        replaceNd(delNode, None)
        dropNode(delNode)


def delt(root, point, depth):
  k = len(point)
  cd = depth % k

  if root is None:
      return
  elif point[cd] < root.point[cd]:
      delt(root.left, point, depth + 1)

  elif point[cd] > root.point[cd]:
      delt(root.right, point, depth + 1)

  else:
      deleteNode(root)
#**----------------------------------------------------**#



#VISUALIZADOR DE ARBOL
#**----------------------------------------------------**#
def viewTree(tree, cont):
  if tree is None:
    return
  else:
    viewTree(tree.right, cont + 1)
    for i in range(cont):
      print(end = "\t \t")
    print(tree.point, "\n")
    viewTree(tree.left, cont + 1)
#**----------------------------------------------------**#

def pointList(arr):
  arr2 = arr[0:1]
  for i in range(len(arr)):
    if arr[i] not in arr2:
        arr2.append(arr[i])
  return arr2


def SED(X, Y):
   return sum((i-j)**2 for i, j in zip(X, Y))

NNRecord = collections.namedtuple("NNRecord", ["point", "distance"])
class KNNpoints:
  def __init__(self, k):
    self.k = k
    self.root = None

  def trainKNN(self, yTrain):
      self.yTrain = np.array(yTrain)

  def predictPoints(self, treeTrain, pointTest):
        X_entrenamiento = []
        
        X_entrenamiento = pointList(search(treeTrain, pointTest, X_entrenamiento))

        distancias = [SED(pointTest, x_entrenamiento) for x_entrenamiento in X_entrenamiento]
        k_indices = np.argsort(distancias)[:self.k]
        k_etiquetas_cercanas = [self.yTrain[i] for i in k_indices]
        etiqueta_predominante = Counter(k_etiquetas_cercanas).most_common(1)[0][0]

        return etiqueta_predominante