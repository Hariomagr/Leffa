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
from diffusers.utils import load_image

import gradio as gr

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

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

        vt_model_dc = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon_dc.pth",
            dtype="float16",
        )
        self.vt_inference_dc = LeffaInference(model=vt_model_dc)

        pt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
            pretrained_model="./ckpts/pose_transfer.pth",
            dtype="float16",
        )
        self.pt_inference = LeffaInference(model=pt_model)

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        mask_image_path,
        control_type,
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False
    ):
        assert control_type in [
            "virtual_tryon", "pose_transfer"], "Invalid control type: {}".format(control_type)
        src_image = load_image(src_image_path)
        ref_image = load_image(ref_image_path)
        mask_image = load_image(mask_image_path)
        src_image = resize_and_center(src_image, 768, 1024).convert("RGB")
        ref_image = resize_and_center(ref_image, 768, 1024).convert("RGB")
        mask_image = resize_and_center(mask_image, 768, 1024).convert("RGB")

        src_image_array = np.array(src_image)

        # DensePose
        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(
                    src_image_array)[:, :, ::-1]
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
            elif vt_model_type == "dress_code":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(
                    src_image_array)
                src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                src_image_seg_array = np.concatenate(
                    [src_image_seg_array] * 3, axis=-1)
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
        elif control_type == "pose_transfer":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(
                src_image_array)[:, :, ::-1]
            src_image_iuv = Image.fromarray(src_image_iuv_array)
            densepose = src_image_iuv

        # Leffa
        transform = LeffaTransform()

        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask_image],
            "densepose": [densepose],
        }
        data = transform(data)
        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                inference = self.vt_inference_hd
            elif vt_model_type == "dress_code":
                inference = self.vt_inference_dc
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
        return gen_image

    def leffa_predict_vt(self, src_image_path, ref_image_path, mask_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint):
        return self.leffa_predict(src_image_path, ref_image_path, mask_image_path, "virtual_tryon", ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint)

    def leffa_predict_pt(self, src_image_path, ref_image_path, mask_image_path, ref_acceleration, step, scale, seed):
        return self.leffa_predict(src_image_path, ref_image_path, mask_image_path, "pose_transfer", ref_acceleration, step, scale, seed)

leffa_predictor = None
def init():
    # Download checkpoints
    snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")
    leffa_predictor = LeffaPredictor()

def runModel(
    cloth_image_url,
    person_image_url,
    mask_image_url,
    acceleration = False,
    step = 50,
    scale = 2.5,
    seed = 42,
    model_type = "viton_hd", # dress_code or viton_hd
    garment_type = "upper_body", # upper_body, lower_body, dresses
    repaint = False
):
    generated_image = leffa_predictor.leffa_predict_vt(
        person_image_url,
        cloth_image_url,
        mask_image_url,
        acceleration,
        step,
        scale,
        seed,
        model_type,
        garment_type,
        repaint
    )
    generated_image.save("generated_image.png")
    return generated_image
