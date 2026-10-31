path = "/home/devraj/MTok/Video-R1/transformers-main/src/transformers/generation/configuration_utils.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

target = "decoder_config_dict = decoder_config.to_dict()"
replacement = "decoder_config_dict = decoder_config.to_dict() if hasattr(decoder_config, 'to_dict') else (decoder_config if isinstance(decoder_config, dict) else {})"

if target in text:
    text = text.replace(target, replacement)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Patched configuration_utils.py successfully!")
else:
    print("Target already patched or not found.")
