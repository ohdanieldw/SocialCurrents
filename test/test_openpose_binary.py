from cv_models.vision.openpose_analyzer import OpenPoseAnalyzer
import os

print("OPENPOSE_BIN:", os.environ.get("OPENPOSE_BIN"))
print("OPENPOSE_MODEL_FOLDER:", os.environ.get("OPENPOSE_MODEL_FOLDER"))
print()

op = OpenPoseAnalyzer()

print("Available:", op.available)
print("Unavailable reason:", getattr(op, "unavailable_reason", "OK"))
