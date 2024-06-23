virtualenv -p $(which python3.8) --system-site-packages peract_env  
source peract_env/bin/activate
pip ipykernel jupyter matplotlib natsort
pip install scipy ftfy regex tqdm torch git+https://github.com/openai/CLIP.git einops pyrender==0.1.45 trimesh==3.9.34 pycollada==0.6