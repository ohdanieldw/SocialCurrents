from cv_models.vision.mediapipe_pose_analyzer import MediaPipePoseAnalyzer

video = "data/sample_person_video.mp4"

print("Initializing MediaPipePoseAnalyzer...")
analyzer = MediaPipePoseAnalyzer()

print("Running inference...")
features = analyzer.get_feature_dict(video)

print("\n=== OUTPUT KEYS ===")
for k in features.keys():
    print(k)

print("\n=== SAMPLE VALUES ===")
for k, v in list(features.items())[:5]:
    print(k, ":", v)
