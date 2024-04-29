"""
Author: Gabe Grand

Tools for running inference of a pretrained ControlNet model.
Adapted from gradio_scribble2image.py from the original authors.

"""


import sys

sys.path.append("..")
from share import *

import einops
import numpy as np
import torch
import random

from PIL import Image
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


A_PROMPT_DEFAULT = "best quality, extremely detailed"
N_PROMPT_DEFAULT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"


def run_sampler(
    model,
    input_image: np.ndarray,
    prompt: str,
    num_samples: int = 1,
    image_resolution: int = 256,
    seed: int = -1,
    a_prompt: str = A_PROMPT_DEFAULT,
    n_prompt: str = N_PROMPT_DEFAULT,
    guess_mode=False,
    strength=1.0,
    ddim_steps=50,
    eta=0.0,
    scale=9.0,
    show_progress: bool = True,
):
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()

        ddim_sampler = DDIMSampler(model)

        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            show_progress=show_progress,
        )

        model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]

        return results

def run_experiments(model, file_path, output_name, prompt=''):
    image_pil = Image.open(file_path)
    image_np = np.array(image_pil)
    results = run_sampler(model, image_np, prompt)
    image = Image.fromarray(results[0])
    image.save(output_name)
    print(f'Image saved successfully: {output_name}')

if __name__ == '__main__':
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_ini9_10__myds.pth', location='cpu'))

    run_experiments(model, 'training/geometricShapes14k/source/327.png', 'infer_image_fl10.png', 'blue flames')
    run_experiments(model, 'training/geometricShapes14k/source/327.png', 'infer_image_fl10_np.png', '')
    run_experiments(model, 'training/geometricShapes14k/source/564.png', 'infer_image_b10.png', 'hole in the bark of a tree')
    run_experiments(model, 'training/geometricShapes14k/source/564.png', 'infer_image_b10_np.png', '')
    run_experiments(model, 'guitar2_p.png', 'infer_image_g10.png', 'man with a  guitar')
    run_experiments(model, 'guitar2_p.png', 'infer_image_g10_np.png', '')



