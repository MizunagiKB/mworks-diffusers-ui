import os
import json
import datetime
import typing

import PIL.Image
import PIL.ImageColor
import PIL.PngImagePlugin

import gradio as gr
import torch
import torchvision

import model.model_types

import model.linaqruf_anything_lpw
import model.linaqruf_anything

LIST_DEVICE = ["cuda", "mps", "cpu"]
DEVICE = None
UI_IMAGE_TXT2IMG = None

IMG_EXPORT_DIR = "./export"

PNG_EMBEDDED_JSON = "mizu_diffusers/embedded_json"


dict_module = {
    model.linaqruf_anything.__name__: model.linaqruf_anything,
    model.linaqruf_anything_lpw.__name__: model.linaqruf_anything_lpw,
}


def callback_processing(step: int, timestep: int, latents: torch.FloatTensor):
    pass


def txt2img_inference(
    module_name: str,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: int,
    num_images_per_prompt: int,
    seed: int,
):

    param = model.model_types.CTxt2Img(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=int(num_images_per_prompt),
        eta=0.0,
        seed=int(seed),
    )

    embedded_data = model.model_types.CEmbeddedData(
        module_name=module_name, param=param
    )

    pipeline_def = dict_module[module_name]

    list_image = pipeline_def.inference_txt2img(param, DEVICE, callback_processing)

    im_result = list_image[0]

    if False:
        png_info = PIL.PngImagePlugin.PngInfo()
        png_info.add_text(PNG_EMBEDDED_JSON, embedded_data.json())

        # Save Image
        dirname = os.path.join(
            IMG_EXPORT_DIR, datetime.datetime.utcnow().strftime("%Y%m%d")
        )
        os.makedirs(dirname)

        pathname = os.path.join(
            dirname, datetime.datetime.utcnow().strftime("%H%M%S.png")
        )
        im_result.save(pathname, "PNG", pnginfo=png_info)

    return im_result


# ----


def txt2img_parse_png_text(temp_filename: str):

    image = PIL.Image.open(temp_filename)

    dict_embedded = json.loads(image.text[PNG_EMBEDDED_JSON])

    embedded_data = model.model_types.CEmbeddedData(**dict_embedded)
    param = typing.cast(model.model_types.CTxt2Img, embedded_data.param)

    return (
        embedded_data.param,
        embedded_data.module_name,
        param.prompt,
        param.negative_prompt,
        param.num_inference_steps,
        param.guidance_scale,
        param.num_images_per_prompt,
        param.seed,
    )


def main():
    global DEVICE, UI_IMAGE_TXT2IMG

    # Check Torch Device
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    # Create Widgets
    ui_device = gr.Radio(LIST_DEVICE, value=DEVICE)
    list_module = [k for k in dict_module.keys()]
    ui_module = gr.Dropdown(list_module, value=list_module[0])
    ui_generate_img = gr.Image(show_label=False, type="pil")
    ui_json = gr.JSON()

    UI_IMAGE_TXT2IMG = ui_generate_img

    with gr.Blocks() as ui:
        gr.HTML("")

        with gr.Row():
            ui_device.render()
            ui_module.render()

        with gr.Row():
            with gr.Tab("Txt2Img"):
                ui_generate_btn = gr.Button("Generate", variant="primary")
                with gr.Row():
                    with gr.Column(scale=20):
                        ui_pos_prompt = gr.TextArea("", label="prompt", lines=6)
                        ui_neg_prompt = gr.TextArea(label="negative_prompt", lines=3)
                        with gr.Row():
                            ui_gen_steps = gr.Number(10, label="num_inference_steps")
                            ui_gen_scale = gr.Number(10.0, label="guidance_scale")
                            ui_gen_image = gr.Slider(
                                label="num_images_per_prompt",
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                            )

                    with gr.Column(scale=80):
                        ui_generate_val = gr.Number(0, label="seed")
                        ui_generate_img.render()

                ui_generate_btn.click(
                    txt2img_inference,
                    inputs=[
                        ui_module,
                        ui_pos_prompt,
                        ui_neg_prompt,
                        ui_gen_steps,
                        ui_gen_scale,
                        ui_gen_image,
                        ui_generate_val,
                    ],
                    outputs=[ui_generate_img],
                    show_progress=True,
                )

                ui_btn_check = gr.UploadButton("Check PNG Information")
                ui_btn_check.upload(
                    txt2img_parse_png_text,
                    inputs=[ui_btn_check],
                    outputs=[
                        ui_json,
                        ui_module,
                        ui_pos_prompt,
                        ui_neg_prompt,
                        ui_gen_steps,
                        ui_gen_scale,
                        ui_gen_image,
                        ui_generate_val,
                    ],
                )
                ui_json.render()

            with gr.Tab("Img2Img"):
                gr.Markdown("x")

            with gr.Tab("Inpaint"):
                gr.Markdown("x")

    ui.launch()


if __name__ == "__main__":
    main()
