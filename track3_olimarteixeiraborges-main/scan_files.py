import os

for root, dirs, files in os.walk('/input'):
    for f in files:
        print(os.path.join(root, f))
