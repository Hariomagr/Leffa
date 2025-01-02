import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

import gradio as gr

# Download checkpoints
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")


class LeffaPredictor(object):
    def __init__(self):
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        vt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
        )
        self.vt_inference = LeffaInference(model=vt_model)
        self.vt_model_type = "viton_hd"

        pt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
            pretrained_model="./ckpts/pose_transfer.pth",
        )
        self.pt_inference = LeffaInference(model=pt_model)

    def change_vt_model(self, vt_model_type):
        if vt_model_type == self.vt_model_type:
            return
        if vt_model_type == "viton_hd":
            pretrained_model = "./ckpts/virtual_tryon.pth"
        elif vt_model_type == "dress_code":
            pretrained_model = "./ckpts/virtual_tryon_dc.pth"
        vt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model=pretrained_model,
        )
        self.vt_inference = LeffaInference(model=vt_model)
        self.vt_model_type = vt_model_type

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=True,
        step=50,
        scale=2.5,
        seed=42,
        vt_garment_type="upper_body",
        vt_repaint=False
    ):
        assert control_type in [
            "virtual_tryon", "pose_transfer"], "Invalid control type: {}".format(control_type)
        src_image = Image.open(src_image_path)
        ref_image = Image.open(ref_image_path)
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        # Mask
        if control_type == "virtual_tryon":
            src_image = src_image.convert("RGB")
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))
            if self.vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(
                    model_parse, keypoints, vt_garment_type)
            elif self.vt_model_type == "dress_code":
                mask = get_agnostic_mask_dc(
                    model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024))
            # garment_type_hd = "upper" if vt_garment_type in [
            #     "upper_body", "dresses"] else "lower"
            # mask = self.mask_predictor(src_image, garment_type_hd)["mask"]
        elif control_type == "pose_transfer":
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)

        # DensePose
        if control_type == "virtual_tryon":
            if self.vt_model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(
                    src_image_array)
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
            elif self.vt_model_type == "dress_code":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(
                    src_image_array)
                src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                src_image_seg_array = np.concatenate(
                    [src_image_seg_array] * 3, axis=-1)
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
        elif control_type == "pose_transfer":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(
                src_image_array)
            src_image_iuv = Image.fromarray(src_image_iuv_array)
            densepose = src_image_iuv

        # Leffa
        transform = LeffaTransform()

        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        if control_type == "virtual_tryon":
            inference = self.vt_inference
        elif control_type == "pose_transfer":
            inference = self.pt_inference
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,)
        gen_image = output["generated_image"][0]
        # gen_image.save("gen_image.png")
        return np.array(gen_image), np.array(mask), np.array(densepose)

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint):
        self.change_vt_model(vt_model_type)
        return self.leffa_predict(src_image_path, ref_image_path, "virtual_tryon", ref_acceleration, step, scale, seed, vt_garment_type, vt_repaint)

    def leffa_predict_pt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed):
        return self.leffa_predict(src_image_path, ref_image_path, "pose_transfer", ref_acceleration, step, scale, seed)


if __name__ == "__main__":

    leffa_predictor = LeffaPredictor()
    example_dir = "./ckpts/examples"
    person1_images = list_dir(f"{example_dir}/person1")
    person2_images = list_dir(f"{example_dir}/person2")
    garment_images = list_dir(f"{example_dir}/garment")

    title = "## Leffa: Learning Flow Fields in Attention for Controllable Person Image Generation"
    link = "[📚 Paper](https://arxiv.org/abs/2412.08486) - [🤖 Code](https://github.com/franciszzj/Leffa) - [🔥 Demo](https://huggingface.co/spaces/franciszzj/Leffa) - [🤗 Model](https://huggingface.co/franciszzj/Leffa)"
    news = """## News
            - 02/Jan/2025, Update the mask generator to improve results. Add ref unet acceleration, boosting prediction speed by 30%. Include more controls in Advanced Options to enhance user experience. Enable intermediate result output for easier development. Enjoy using it!.
            More news can be found in the [GitHub repository](https://github.com/franciszzj/Leffa).
            """
    description = "Leffa is a unified framework for controllable person image generation that enables precise manipulation of both appearance (i.e., virtual try-on) and pose (i.e., pose transfer)."
    note = "Note: The models used in the demo are trained solely on academic datasets. Virtual try-on uses VITON-HD/DressCode, and pose transfer uses DeepFashion."

    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.pink, secondary_hue=gr.themes.colors.red)).queue() as demo:
        gr.Markdown(title)
        gr.Markdown(link)
        gr.Markdown(news)
        gr.Markdown(description)

        with gr.Tab("Control Appearance (Virtual Try-on)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Person Image")
                    vt_src_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Person Image",
                        width=512,
                        height=512,
                    )

                    gr.Examples(
                        inputs=vt_src_image,
                        examples_per_page=10,
                        examples=person1_images,
                    )

                with gr.Column():
                    gr.Markdown("#### Garment Image")
                    vt_ref_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Garment Image",
                        width=512,
                        height=512,
                    )

                    gr.Examples(
                        inputs=vt_ref_image,
                        examples_per_page=10,
                        examples=garment_images,
                    )

                with gr.Column():
                    gr.Markdown("#### Generated Image")
                    vt_gen_image = gr.Image(
                        label="Generated Image",
                        width=512,
                        height=512,
                    )

                    with gr.Row():
                        vt_gen_button = gr.Button("Generate")

                    with gr.Accordion("Advanced Options", open=False):
                        vt_model_type = gr.Radio(
                            label="Model Type",
                            choices=[("VITON-HD (Recommended)", "viton_hd"),
                                     ("DressCode (Experimental)", "dress_code")],
                            value="viton_hd",
                        )

                        vt_garment_type = gr.Radio(
                            label="Garment Type",
                            choices=[("Upper", "upper_body"),
                                     ("Lower", "lower_body"),
                                     ("Dress", "dresses")],
                            value="upper_body",
                        )

                        vt_ref_acceleration = gr.Radio(
                            label="Accelerate Reference UNet",
                            choices=[("True", True), ("False", False)],
                            value=True,
                        )

                        vt_repaint = gr.Radio(
                            label="Repaint Mode",
                            choices=[("True", True), ("False", False)],
                            value=False,
                        )

                        vt_step = gr.Number(
                            label="Inference Steps", minimum=30, maximum=100, step=1, value=50)

                        vt_scale = gr.Number(
                            label="Guidance Scale", minimum=0.1, maximum=5.0, step=0.1, value=2.5)

                        vt_seed = gr.Number(
                            label="Random Seed", minimum=-1, maximum=2147483647, step=1, value=42)

                    with gr.Accordion("Debug", open=False):
                        vt_mask = gr.Image(
                            label="Generated Mask",
                            width=256,
                            height=256,
                        )

                        vt_densepose = gr.Image(
                            label="Generated DensePose",
                            width=256,
                            height=256,
                        )

                vt_gen_button.click(fn=leffa_predictor.leffa_predict_vt, inputs=[
                    vt_src_image, vt_ref_image, vt_ref_acceleration, vt_step, vt_scale, vt_seed, vt_model_type, vt_garment_type, vt_repaint], outputs=[vt_gen_image, vt_mask, vt_densepose])

        with gr.Tab("Control Pose (Pose Transfer)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Person Image")
                    pt_ref_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Person Image",
                        width=512,
                        height=512,
                    )

                    gr.Examples(
                        inputs=pt_ref_image,
                        examples_per_page=10,
                        examples=person1_images,
                    )

                with gr.Column():
                    gr.Markdown("#### Target Pose Person Image")
                    pt_src_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Target Pose Person Image",
                        width=512,
                        height=512,
                    )

                    gr.Examples(
                        inputs=pt_src_image,
                        examples_per_page=10,
                        examples=person2_images,
                    )

                with gr.Column():
                    gr.Markdown("#### Generated Image")
                    pt_gen_image = gr.Image(
                        label="Generated Image",
                        width=512,
                        height=512,
                    )

                    with gr.Row():
                        pose_transfer_gen_button = gr.Button("Generate")

                    with gr.Accordion("Advanced Options", open=False):
                        pt_ref_acceleration = gr.Radio(
                            label="Accelerate Reference UNet",
                            choices=[("True", True), ("False", False)],
                            value=True,
                        )

                        pt_step = gr.Number(
                            label="Inference Steps", minimum=30, maximum=100, step=1, value=50)

                        pt_scale = gr.Number(
                            label="Guidance Scale", minimum=0.1, maximum=5.0, step=0.1, value=2.5)

                        pt_seed = gr.Number(
                            label="Random Seed", minimum=-1, maximum=2147483647, step=1, value=42)

                    with gr.Accordion("Debug", open=False):
                        pt_mask = gr.Image(
                            label="Generated Mask",
                            width=256,
                            height=256,
                        )

                        pt_densepose = gr.Image(
                            label="Generated DensePose",
                            width=256,
                            height=256,
                        )

                pose_transfer_gen_button.click(fn=leffa_predictor.leffa_predict_pt, inputs=[
                    pt_src_image, pt_ref_image, pt_ref_acceleration, pt_step, pt_scale, pt_seed], outputs=[pt_gen_image, pt_mask, pt_densepose])

        gr.Markdown(note)

        demo.launch(share=True, server_port=7860,
                    allowed_paths=["./ckpts/examples"])
