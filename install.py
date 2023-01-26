import launch

if not launch.is_installed("jax"):
    launch.run_pip("install jax[cpu]", "requirements for Converter Face extension")
if not launch.is_installed("flax"):
    launch.run_pip("install flax", "requirements for Converter extension")
if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers", "requirements for Converter extension")
if not launch.is_installed("accelerate"):
    launch.run_pip("install accelerate", "requirements for Converter extension")  
if not launch.is_installed("transformers"):
    launch.run_pip("install transformers", "requirements for Converter extension")
if not launch.is_installed("ftfy"):
    launch.run_pip("install ftfy", "requirements for Converter extension")
if not launch.is_installed("OmegaConf"):
    launch.run_pip("install OmegaConf", "requirements for Converter extension")
if not launch.is_installed("huggingface_hub"):
    launch.run_pip("install huggingface-hub", "requirements for Converter extension")
if not launch.is_installed("safetensors"):
    launch.run_pip("install safetensors", "requirements for Converter extension")
if not launch.is_installed("gdown"):
    launch.run_pip("install gdown", "requirements for Converter extension")
if not launch.is_installed("pytorch_lightning"):
    launch.run_pip("install pytorch_lightning", "requirements for Converter extension")
