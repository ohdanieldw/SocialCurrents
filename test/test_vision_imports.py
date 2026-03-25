from cv_models.vision import vision_status
import pprint

print("\n=== VISION STATUS ===\n")
status = vision_status()
pprint.pprint(status)

print("\n=== FAILED ANALYZERS ===\n")
failed = {k: v for k, v in status.items() if v is not True}
pprint.pprint(failed)
