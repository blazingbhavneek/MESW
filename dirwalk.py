import os

folder = "./graphrag_cache"

for root, dirs, files in os.walk(folder):
    for d in dirs:
        print(os.path.join(root, d))
    for f in files:
        print(os.path.join(root, f))
