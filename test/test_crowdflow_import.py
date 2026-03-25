from cv_models.vision.crowdflow_analyzer import CrowdFlowAnalyzer

c = CrowdFlowAnalyzer()

print("Available:", c.available)
print("Unavailable reason:", getattr(c, "unavailable_reason", "OK"))
