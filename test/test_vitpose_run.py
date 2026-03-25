from cv_models.vision.vitpose_analyzer import ViTPoseAnalyzer

video = "data/sample_person_video.mp4"

v = ViTPoseAnalyzer(device="cpu")

print("Running ViTPose inference...")
out = v.get_feature_dict(video)

print("\n=== OUTPUT ===")
for k, v in out.items():
    print(k, ":", v)
