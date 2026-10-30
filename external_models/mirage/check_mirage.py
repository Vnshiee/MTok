import os
import glob

print("--- Mirage Paper Source ---")
tex_files = glob.glob("/home/devraj/MTok/external_models/mirage/paper_src/**/*.tex", recursive=True)
print(f"Total .tex files: {len(tex_files)}")
for t in tex_files[:10]:
    print(" ", os.path.basename(t))

print("\n--- Mirage Repo Structure ---")
py_files = glob.glob("/home/devraj/MTok/external_models/mirage/repo/**/*.py", recursive=True)
print(f"Total .py files: {len(py_files)}")
for p in py_files[:15]:
    print(" ", os.path.relpath(p, "/home/devraj/MTok/external_models/mirage/repo"))
