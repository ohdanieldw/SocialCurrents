from cv_models.vision.avhubert_analyzer import AVHuBERTAnalyzer

a = AVHuBERTAnalyzer()

print("Available:", a.available)
print("Unavailable reason:", getattr(a, "unavailable_reason", "OK"))
