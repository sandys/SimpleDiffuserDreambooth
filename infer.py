#@title Run for generating images.

import os
os.environ['LD_LIBRARY_PATH']='/usr/lib/x86_64-linux-gnu/:/opt/conda/lib/'

import os
import shutil
def reset_folder(path):
    if os.path.isdir(path):
      shutil.rmtree(path)
    os.makedirs(path,exist_ok=True)
    
#reset_folder("images")
    
MODEL_NAME = "runwayml/stable-diffusion-v1-5" #@param {type:"string"}
#MODEL_NAME = "CompVis/stable-diffusion-v1-4"  #@param {type:"string"}

INSTANCE = "sukh02"#@param {type:"string"}
#INSTANCE_DIR = "/content/data/"+INSTANCE 
INSTANCE_DIR = "data/zwx"


CLASS = "person" #@param {type:"string"}
#CLASS_DIR = "/content/data/"+CLASS
CLASS_DIR = "data/person"


#OUTPUT_DIR = "/content/stable_diffusion_weights/" + INSTANCE
OUTPUT_DIR = "data/stable_diffusion_weights/zwx2" #@param {type:"string"}



import torch

from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline, DPMSolverMultistepScheduler,EulerDiscreteScheduler

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from IPython.display import display

from natsort import natsorted
from glob import glob
import os
model_path = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]  # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive          

# unet_model = UNet2DConditionModel(
#     sample_size=512 // 8,
#     in_channels=4,
#     out_channels=4,
#     down_block_types=("DownBlock2D","DownBlock2D","CrossAttnDownBlock2D","DownBlock2D"),
#     up_block_types=("CrossAttnUpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"),
#     block_out_channels=(320,640,1280,1280),
#     layers_per_block=2,
#     cross_attention_dim=768,
#     attention_head_dim=8,
#     use_linear_projection=False
# )


import random
seed_value= 0
random.seed(seed_value)
g_cuda = torch.Generator(device='cuda').manual_seed(seed_value)

#scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
#scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
#scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")




negative_prompt = ""
num_samples = 6 #@param {type:"number"}
guidance_scale = 10 #@param {type:"number"}
num_inference_steps = 30 #@param {type:"number"}
height = 512 #@param {type:"number"}
width = 512 #@param {type:"number"}

with autocast("cuda"), torch.inference_mode():
    pipe =  DiffusionPipeline.from_pretrained(model_path, force_download=True,safety_checker=None,custom_pipeline="lpw_stable_diffusion", scheduler=scheduler,  torch_dtype=torch.float16).to("cuda")

    images = pipe(
        p3d,
        height=height,
        width=width,
        negative_prompt=neg_p3d,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        #generator=g_cuda
    ).images

import datetime
import time
folder_time = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
for i, img in enumerate(images):
    img.save('images/'+str(i).zfill(4)+folder_time+'.jpg')
    display(img)
torch.cuda.empty_cache()