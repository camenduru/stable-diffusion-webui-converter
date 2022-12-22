import os
import numpy as np
import gradio as gr
from diffusers import FlaxStableDiffusionPipeline
from huggingface_hub import model_info, create_repo, create_branch, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from modules import scripts, script_callbacks

ckpt_file = os.path.join(scripts.basedir(), "", "model.ckpt")
ckpt_path = os.path.join(scripts.basedir(), "ckpt")
pt_path = os.path.join(scripts.basedir(), "pt")
flax_path = os.path.join(scripts.basedir(), "flax")
safetensors_path = os.path.join(scripts.basedir(), "safetensors")

def download_ckpt(ckpt_url):
    os.system(f"wget {ckpt_url} -O {ckpt_file} --no-check-certificate")
    return "download ckpt done!"

def to_pt():
    os.system("wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_original_stable_diffusion_to_diffusers.py --no-check-certificate")
    os.system(f"python3 convert_original_stable_diffusion_to_diffusers.py --checkpoint_path {ckpt_file} --dump_path {pt_path}")
    return "convert to pt done!"

def push_pt(model_to, token, branch):
    try:
        repo_exists = True
        r_info = model_info(model_to, token=token)
    except RepositoryNotFoundError:
        repo_exists = False
    finally:
        if repo_exists:
            print(r_info)
        else:
            create_repo(model_to, private=True, token=token)
    try:
        branch_exists = True
        b_info = model_info(model_to, revision=branch, token=token)
    except RevisionNotFoundError:
        branch_exists = False
    finally:
        if branch_exists:
            print(b_info)
        else:
            create_branch(model_to, branch=branch, token=token)
    upload_folder(folder_path=pt_path, path_in_repo="", revision=branch, repo_id=model_to, commit_message=f"pt", token=token)
    return "push pt done!"
    
def delete_pt():
    os.system(f"rm -rf {pt_path}")
    return "delete pt done!"
    
def clone_pt(model_url):
    os.system("git lfs install")
    os.system(f"git clone https://huggingface.co/{model_url} {pt_path}")
    return "clone pt done!"
    
def to_flax():
    pipe, params = FlaxStableDiffusionPipeline.from_pretrained(pt_path, from_pt=True)
    pipe.save_pretrained(flax_path, params=params)
    return "convert to flax done!"

def push_flax(model_to, token, branch):
    try:
        repo_exists = True
        r_info = model_info(model_to, token=token)
    except RepositoryNotFoundError:
        repo_exists = False
    finally:
        if repo_exists:
            print(r_info)
        else:
            create_repo(model_to, private=True, token=token)
    try:
        branch_exists = True
        b_info = model_info(model_to, revision=branch, token=token)
    except RevisionNotFoundError:
        branch_exists = False
    finally:
        if branch_exists:
            print(b_info)
        else:
            create_branch(model_to, branch=branch, token=token)
    upload_folder(folder_path=flax_path, path_in_repo="", revision=branch, repo_id=model_to, commit_message=f"flax", token=token)
    return "push flax done!"

def delete_flax():
    os.system(f"rm -rf {flax_path}")
    return "delete flax done!"
    
def to_ckpt(ckpt_name):
    os.system("wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_diffusers_to_original_stable_diffusion.py --no-check-certificate")
    os.system(f"mkdir {ckpt_path}")
    checkpoint_path = os.path.join(ckpt_path, "", f"{ckpt_name}.ckpt")
    os.system(f"python3 convert_diffusers_to_original_stable_diffusion.py --model_path {pt_path} --checkpoint_path {checkpoint_path}")
    return "convert to ckpt done!"

def push_ckpt(model_to, token, branch):
    try:
        repo_exists = True
        r_info = model_info(model_to, token=token)
    except RepositoryNotFoundError:
        repo_exists = False
    finally:
        if repo_exists:
            print(r_info)
        else:
            create_repo(model_to, private=True, token=token)
    try:
        branch_exists = True
        b_info = model_info(model_to, revision=branch, token=token)
    except RevisionNotFoundError:
        branch_exists = False
    finally:
        if branch_exists:
            print(b_info)
        else:
            create_branch(model_to, branch=branch, token=token)    
    upload_folder(folder_path=ckpt_path, path_in_repo="", revision=branch, repo_id=model_to, commit_message=f"ckpt", token=token)
    return "push ckpt done!"
    
def delete_ckpt():
    os.system(f"rm -rf {ckpt_path}")
    return "delete ckpt done!"
    
def to_safetensors(safetensors_name):
    weights = torch.load(ckpt_file)["state_dict"]
    if "state_dict" in weights:
        weights.pop("state_dict")
        for k, v in weights.items():
          print(k, type(v))
    # os.system("wget -q https://raw.githubusercontent.com/huggingface/safetensors/main/bindings/python/convert.py")
    os.system(f"mkdir {safetensors_path}")
    safetensors_file = os.path.join(safetensors_path, "", f"{safetensors_name}.safetensors")
    save_file(weights, safetensors_file)
    return "convert to safetensors done!"

def push_safetensors(model_to, token, branch):
    try:
        repo_exists = True
        r_info = model_info(model_to, token=token)
    except RepositoryNotFoundError:
        repo_exists = False
    finally:
        if repo_exists:
            print(r_info)
        else:
            create_repo(model_to, private=True, token=token)
    try:
        branch_exists = True
        b_info = model_info(model_to, revision=branch, token=token)
    except RevisionNotFoundError:
        branch_exists = False
    finally:
        if branch_exists:
            print(b_info)
        else:
            create_branch(model_to, branch=branch, token=token)
    upload_folder(folder_path=safetensors_path, path_in_repo="", revision=branch, repo_id=model_to, commit_message=f"safetensors", token=token)
    return "push safetensors done!"

def delete_safetensors():
    os.system(f"rm -rf {safetensors_path}")
    return "delete safetensors done!"

def on_ui_tabs():     
    with gr.Blocks() as converter:
        gr.Markdown(
        """
        ### ckpt to pytorch
        ckpt_url = https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.ckpt <br />
        pt_model_to = camenduru/openjourney <br />
        branch = main <br />
        token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_ckpt_url = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_url")
                    text_pt_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="pt_model_to")
                    text_pt_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="branch")
                    text_pt_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                    out_pt = gr.Textbox(show_label=False)
                with gr.Row().style(equal_height=True):
                    btn_download_ckpt = gr.Button("Download CKPT")
                    btn_to_pt = gr.Button("Convert to PT")
                    btn_push_pt = gr.Button("Push PT to 🤗")
                    btn_delete_pt = gr.Button("Delete PT")
            btn_download_ckpt.click(download_ckpt, inputs=[text_ckpt_url], outputs=out_pt)
            btn_to_pt.click(to_pt, outputs=out_pt)
            btn_push_pt.click(push_pt, inputs=[text_pt_model_to, text_pt_token, text_pt_branch], outputs=out_pt)
            btn_delete_pt.click(delete_pt, outputs=out_pt)
        gr.Markdown(
        """
        ### pytorch to flax <br />
        pt_model_from = prompthero/openjourney <br />
        flax_model_to = camenduru/openjourney <br />
        branch = flax <br />
        token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write <br />
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_pt_model_from = gr.Textbox(show_label=False, max_lines=1, placeholder="pt_model_from")
                    text_flax_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="flax_model_to")
                    text_flax_branch = gr.Textbox(show_label=False, value="flax", max_lines=1, placeholder="flax_branch")
                    text_flax_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                    out_flax = gr.Textbox(show_label=False)
                with gr.Row().style(equal_height=True):
                    btn_clone_pt = gr.Button("Clone PT from 🤗")
                    btn_to_flax = gr.Button("Convert to Flax")
                    btn_push_flax = gr.Button("Push Flax to 🤗")
                    btn_delete_flax = gr.Button("Delete Flax")
            btn_clone_pt.click(clone_pt, inputs=[text_pt_model_from], outputs=out_flax)
            btn_to_flax.click(to_flax, outputs=out_flax)
            btn_push_flax.click(push_flax, inputs=[text_flax_model_to, text_flax_token, text_flax_branch], outputs=out_flax)
            btn_delete_flax.click(delete_flax, outputs=out_flax)
        gr.Markdown(
        """
        ### pytorch to ckpt
        pt_model_from = prompthero/openjourney <br />
        ckpt_name = openjourney <br />
        ckpt_model_to = camenduru/openjourney <br />
        branch = ckpt <br />
        token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_pt_model_from = gr.Textbox(show_label=False, max_lines=1, placeholder="pt_model_from")
                    text_ckpt_name = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_name")
                    text_ckpt_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_model_to")
                    text_ckpt_branch = gr.Textbox(show_label=False, value="ckpt", max_lines=1, placeholder="ckpt_branch")
                    text_ckpt_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                    out_ckpt = gr.Textbox(show_label=False)
                with gr.Row().style(equal_height=True):
                    btn_clone_pt = gr.Button("Clone PT from 🤗")
                    btn_to_ckpt = gr.Button("Convert to CKPT")
                    btn_push_ckpt = gr.Button("Push CKPT to 🤗")
                    btn_delete_ckpt = gr.Button("Delete CKPT")
            btn_clone_pt.click(clone_pt, inputs=[text_pt_model_from], outputs=out_ckpt)
            btn_to_ckpt.click(to_ckpt, inputs=[text_ckpt_name], outputs=out_ckpt)
            btn_push_ckpt.click(push_ckpt, inputs=[text_ckpt_model_to, text_ckpt_token, text_ckpt_branch], outputs=out_ckpt)
            btn_delete_ckpt.click(delete_ckpt, outputs=out_ckpt)
        gr.Markdown(
        """
        ### ckpt to safetensors <br />
        ckpt_url = https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.ckpt <br />
        safetensors_name = openjourney <br />
        safetensors_model_to = camenduru/openjourney <br />
        branch = safetensors <br />
        token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write <br />
        """)
        with gr.Group():
            with gr.Box():
                with gr.Row().style(equal_height=True):
                    text_ckpt_url = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_url")
                    text_safetensors_name = gr.Textbox(show_label=False, max_lines=1, placeholder="safetensors_name")
                    text_safetensors_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="safetensors_model_to")
                    text_safetensors_branch = gr.Textbox(show_label=False, value="safetensors", max_lines=1, placeholder="safetensors_branch")
                    text_safetensors_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                    out_safetensors = gr.Textbox(show_label=False)
                with gr.Row().style(equal_height=True):
                    btn_download_ckpt = gr.Button("Download CKPT")
                    btn_to_safetensors = gr.Button("Convert to safetensors")
                    btn_push_safetensors = gr.Button("Push safetensors to 🤗")
                    btn_delete_safetensors = gr.Button("Delete safetensors")
            btn_download_ckpt.click(download_ckpt, inputs=[text_ckpt_url], outputs=out_safetensors)
            btn_to_safetensors.click(to_safetensors, inputs=[text_safetensors_name], outputs=out_safetensors)
            btn_push_safetensors.click(push_safetensors, inputs=[text_safetensors_model_to, text_safetensors_token, text_safetensors_branch], outputs=out_safetensors)
            btn_delete_safetensors.click(delete_safetensors, outputs=out_safetensors)
    return (converter, "Converter", "converter"),
script_callbacks.on_ui_tabs(on_ui_tabs)
