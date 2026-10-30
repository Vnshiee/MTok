path = "/home/devraj/MTok/external_models/lvr/STEP1_INGESTION_MANIFEST.md"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix 1: output_dir on line 462
text = text.replace(
    '`output_dir: "/eb_data/bangzheng/output_models/Stage1_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128"`',
    '`output_dir: "/dockerx/Local/users/bangzheng/Qwen2.5-VL-7B-Instruct/Stage1_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128-3ug6bgwn"`'
)

# Fix 2: training_args.bin line in table
text = text.replace(
    '| [`LVR-7B/training_args.bin`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/training_args.bin) | 53 | 54 | 9,425 |',
    '| [`LVR-7B/training_args.bin`](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/training_args.bin) | N/A (Binary) | N/A (Binary) | 9,425 |'
)

# Fix 3: config.json line numbers and keys in Section 4.3 A
text = text.replace(
    '- **Special Config Parameters** ([L24-L35](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L24-L35)):',
    '- **Special Config Parameters** ([L13-L20](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L13-L20)):'
)
text = text.replace(
    '  - `lvr_head_type`: `"mlp"` ([L32](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L32))',
    '  - `lvr_head_type`: `"simple"` ([L17](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L17))'
)
text = text.replace(
    '`transformers_version: "4.54.0"` ([L103](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L103))',
    '`transformers_version: "4.54.1"` ([L110](file:///home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B/config.json#L110))'
)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Applied clean final fixes.")
