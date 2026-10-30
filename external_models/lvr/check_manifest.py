import re

with open("/home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md", "r") as f:
    text = f.read()

print("File size:", len(text))
for sec in ["Section 1", "Section 2", "Section 3", "Section 4"]:
    print(sec, "found:", sec in text)

