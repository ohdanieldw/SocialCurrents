from cv_models.vision.crowdflow_analyzer import CrowdFlowAnalyzer

video = "data/sample_person_video.mp4"

c = CrowdFlowAnalyzer()

if not c.available:
    raise RuntimeError(c.unavailable_reason)

print("Running CrowdFlow (this may be slow)...")
out = c.get_feature_dict(video)

print("\n=== OUTPUT ===")
for k, v in out.items():
    print(k, ":", v)
