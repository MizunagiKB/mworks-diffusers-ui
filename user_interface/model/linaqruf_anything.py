import typing

import PIL.Image
import torch

from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
)

import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion as txt2img
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img as img2iimg

import model.model_types


MODEL_NAME = "Linaqruf/anything-v3.0"


def false_safety(images, **kwargs):
    return images, False


def inference_txt2img(
    param: model.model_types.CTxt2Img,
    device: str,
    callback: typing.Optional[
        typing.Callable[[int, int, torch.FloatTensor], None]
    ] = None,
    is_cancelled_callback: typing.Optional[typing.Callable[[], bool]] = None,
    callback_steps: typing.Optional[int] = 1,
) -> list[PIL.Image.Image]:

    pipe = typing.cast(
        txt2img.StableDiffusionPipeline,
        StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            # custom_pipeline="lpw_stable_diffusion",
            torch_dtype=torch.float16,
        ).to(device),
    )

    pipe.enable_attention_slicing()
    pipe.safety_checker = false_safety

    generator = torch.Generator().manual_seed(int(param.seed))

    return pipe(
        param.prompt,
        height=param.image_h,
        width=param.image_w,
        num_inference_steps=param.num_inference_steps,
        guidance_scale=param.guidance_scale,
        negative_prompt=param.negative_prompt,
        num_images_per_prompt=param.num_images_per_prompt,
        max_embeddings_multiples=3,
        eta=param.eta,
        generator=generator,
    ).images


def inference_img2img(
    image: PIL.Image.Image,
    param: model.model_types.CImg2Img,
    device: str,
    callback: typing.Optional[
        typing.Callable[[int, int, torch.FloatTensor], None]
    ] = None,
    is_cancelled_callback: typing.Optional[typing.Callable[[], bool]] = None,
    callback_steps: typing.Optional[int] = 1,
) -> list[PIL.Image.Image]:

    return [PIL.Image.new("RGB", size=image.size)]
