import os
import glob
print(f"Propagating cwd: {os.getcwd()}", flush=True)
print(f"Files in dir: {os.listdir('.')}", flush=True)
print(f"Glob result for sample_image4.png: {glob.glob('sample_image4.png')}", flush=True)
print(f"Glob result for *.png: {glob.glob('*.png')}", flush=True)
