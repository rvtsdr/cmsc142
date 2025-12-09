# UPDATE: ok na yung frontend, waiting na lang for the APIs for:
# get initial map
# get solution from the algo

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID, uuid4

import json

import numpy as np
import random

app = FastAPI()

map1 = np.array([
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,-1,0,0,0,0,0,0,0],
    [-1,-1,0,0,0,0,0,0,0],
    [0,-1,0,0,-1,-1,-1,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,-1,0,0,0,0,0],
    [0,0,0,0,0,0,0,-1,0],
    [0,0,0,0,0,-1,0,0,0]
])

map2 = np.array([
    [0,-1,0,0,0,0,0,0,0],
    [0,0,-1,0,0,0,0,0,0],
    [0,0,0,0,0,-1,-1,0,0],
    [0,-1,-1,0,0,0,0,0,0],
    [0,0,0,0,0,-1,-1,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,-1,0,0,0,-1,0]
])

map3 = np.array([
    [0, -1, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, -1, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 0, 0]
])

maps = {
    1: map1,
    2: map2,
    3: map3
}

################################

def easy_init():
    """Resets all game state so each API call starts fresh."""
    global n, m, bombs, x, y
    global map, revealedMap, advancedMap
    global seq, sequence, move, moveBit, target_coor, flag, click, steps

    n = 9
    m = 9
    bombs = 10
    x = 0
    y = 8  # opening move

    preset = random.choice([1, 2, 3])
    map = maps[preset].copy() 

    revealedMap = np.full((n, m), "x", dtype=object)
    advancedMap = np.full((n, m), "x", dtype=object)

    seq = [revealedMap.copy()] 
    sequence = [revealedMap.copy()]  # timeline of revealed maps
    click, flag = 0, 1
    move = []                        # text like "flag (2,3)"
    moveBit = []                     # 0 = click, 1 = flag
    target_coor = []                      # coordinate pairs

################################

#THIS CELL IS FOR FUNCTIONS ONLY

def neighbors(x, y, matrix, c):          #Returns number of neighbors of matrix[x, y] equal to c
  rowMin = max(0, x-1)                   #Declare valid bounds for neighbors
  rowMax = min(n, x+2)
  colMin = max(0, y-1)
  colMax = min(m, y+2)

  count = np.sum(matrix[rowMin:rowMax, colMin:colMax] == c)

  return count, rowMin, rowMax, colMin, colMax

def floodFill(x, y):
  if revealedMap[x, y] != 'x':            #If tile has been opened, stop flood fill
      return
                                          #Count num of neighbors with bombs
  countB, rowMin, rowMax, colMin, colMax = neighbors(x, y, map, -1)

  if countB > 0:                          #Stop flood fill when count > 0
    revealedMap[x, y] = countB

  else:
    revealedMap[x, y] = 'o'                 #Continue flood fill until tile with
                                                        #count/number is found
    for i in range(rowMin, rowMax):
      for j in range(colMin, colMax):
        floodFill(i, j)

  return

def revealTiles(x, y):
  if map[x, y] == -1:            #If the opened tile is a bomb,
    revealedMap[x, y] = -1       #reveal that tile,
    return -1                    #declare game over.

  else:                          #Else, proceed with Flood Fill
    floodFill(x, y)
    return 0

def flagMines(x, y):
  countX, rowMin, rowMax, colMin, colMax = neighbors(x, y, revealedMap, 'x')
  countM = np.sum(revealedMap[rowMin:rowMax, colMin:colMax] == 'm')

  if revealedMap[x, y] == (countX + countM):
    for i in range(rowMin, rowMax):
      for j in range(colMin, colMax):
        if revealedMap[i, j] == 'x':
          revealedMap[i, j] = 'm'
          countM +=1
          countX -=1
          move.append(f"Flag ({i}, {j})")
          moveBit.append(flag)
          target_coor.append([i, j])
          sequence.append(revealedMap.copy())

  if (revealedMap[x, y] == countM) & (countX != 0):
        move.append(f"Click on ({x}, {y})")
        moveBit.append(click)
        target_coor.append([x,y])

        for i in range(rowMin, rowMax):
          for j in range(colMin, colMax):
            floodFill(i, j)

        sequence.append(revealedMap.copy())

  return

def solver(bombs, flags, k):                 #k tracks progress of solver
  for i in range(n):                         #Iterate through the whole map
    for j in range(m):                       #If a number is encountered,
      if revealedMap[i, j] not in ('o', 'x', 'm'):
        flagMines(i, j)

  currentFlags = np.sum(revealedMap == 'm')  #Count number of flags in the whole map
  k = {True: k+1, False: 0}[currentFlags == flags]        #Update k

  if k==2:
    #advancedSolver()
    print(f"Switching to Advanced Solver")
    return

  if bombs != currentFlags:
    print(f"\nNumber of flags is {currentFlags}")
    print(f"k is {k}")
    solver(bombs, currentFlags, k)

def main():
  global seq
  revealTiles(x, y)
  solver(bombs, 0, 0)
  
  seq = [[[str(elem) for elem in row] for row in arr.tolist()] for arr in sequence]
  print(len(seq))
  print(len(moveBit))
  print(len(target_coor))

easy_init()
main()

################################

class CombinedStep(BaseModel):
    step: List[List[str]]
    moveBit: int
    target: List[int]

class CombinedSolution(BaseModel):
    combined: List[CombinedStep]

combined = [
    {
        "step": seq[i],
        "moveBit": moveBit[i],
        "target": target_coor[i]
    }
    for i in range(len(moveBit))
]

@app.get("/easy_map/")
async def get_easy_map():
    easy_init()
    return map.tolist()

class Solution(BaseModel):
    steps: List[List[List[str]]]
    moveBit: List[int]
    target: List[List[int]]

@app.get("/easy_solution/")
async def get_easy_solution():
    easy_init()
    main()

    combined = [
        {
            "step": seq[i],
            "moveBit": moveBit[i],
            "target": target_coor[i]
        }
        for i in range(len(moveBit))
    ]

    combined.append(seq[len(seq)-1])

    return {"combined": combined}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)