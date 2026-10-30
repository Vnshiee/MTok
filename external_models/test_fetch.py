import urllib.request
import json

targets = [
    ("Monet", "https://api.github.com/repos/NOVAglow646/Monet", "https://arxiv.org/abs/2511.21395"),
    ("Mirage", "https://api.github.com/repos/UMass-Embodied-AGI/Mirage", "https://arxiv.org/abs/2506.17218"),
]

req = urllib.request.Request("https://api.github.com/repos/NOVAglow646/Monet", headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        print("GitHub Monet:", resp.status)
except Exception as e:
    print("GitHub Monet error:", e)

req2 = urllib.request.Request("https://api.github.com/repos/UMass-Embodied-AGI/Mirage", headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req2) as resp:
        print("GitHub Mirage:", resp.status)
except Exception as e:
    print("GitHub Mirage error:", e)
