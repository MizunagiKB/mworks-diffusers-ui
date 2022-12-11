import typing
import PIL.Image
from pydantic import BaseModel


PNG_EMBED_MODEL = "mizu_diffusers/model_name"
PNG_EMBED_MODULE = "mizu_diffusers/module"
PNG_EMBED_PARAM = "mizu_diffusers/param"


class CTxt2Img(BaseModel):
    prompt: str
    image_w: int = 512
    image_h: int = 512
    num_inference_steps: int = 1
    guidance_scale: float = 10.0
    negative_prompt: typing.Union[str, None] = None
    num_images_per_prompt: int = 1
    eta: float = 0.0
    seed: int = 0.0


class CImg2Img(BaseModel):
    prompt: str
    negative_prompt: typing.Union[str, None] = None
    strength: float = 0.75
    num_inference_steps: int = 1
    guidance_scale: float = 10.0
    num_images_per_prompt: int = 1
    eta: float = 0.0
    seed: int = 0.0


class CInpaint(BaseModel):
    prompt: str


class CEmbeddedData(BaseModel):
    module_name: str
    param: typing.Union[CTxt2Img, CImg2Img, CInpaint]
