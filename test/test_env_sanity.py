import sys

try:
    import torch
    import cv2
except Exception as e:
    print("IMPORT ERROR:", e)
    raise

print("Python version:")
print(sys.version)
print()

print("Torch version:")
print(torch.__version__)
print()

print("OpenCV version:")
print(cv2.__version__)
