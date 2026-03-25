from cv_models.vision.openpose_analyzer import OpenPoseAnalyzer

video = "data/sample_person_video.mp4"

op = OpenPoseAnalyzer()

if not op.available:
    raise RuntimeError(op.unavailable_reason)

print("Running OpenPose...")
out = op.get_feature_dict(video)

print("\n=== OUTPUT ===")
for k, v in out.items():
    print(k, ":", v)
