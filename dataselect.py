import os
import random

img_path = []

for root, dir, file in os.walk("./test2017"):
  for name in file:
    img_path.append(str(os.path.join(root, name)))

print(len(img_path))

to_delete = [img_path[i] for i in random.sample(range(0,len(img_path)), len(img_path) - 5000)] 

print(len(to_delete))

for path in to_delete:
  os.remove(path)