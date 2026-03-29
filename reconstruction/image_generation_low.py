import os
from PIL import Image
import torch
import time
import numpy as np
import torch.nn as nn
import pandas as pd
import random
import argparse
try:
    from .custom_pipeline_low import Generator4Embeds_latent2img
    from .model import PriorNetwork, BrainDiffusionPrior, vae_latent_projector
except ImportError:
    from custom_pipeline_low import Generator4Embeds_latent2img
    from model import PriorNetwork, BrainDiffusionPrior, vae_latent_projector

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
parser.add_argument('--test_index_path', default='./TVSD/things_imgs_test.csv', type=str)
parser.add_argument('--models_root', default='./models', type=str)
parser.add_argument('--vae_latent_root', default='./data/TVSD/DNN_feature_maps/vae_latents', type=str)
parser.add_argument('--test_idx_path', default='./TVSD/test_idx_intrain.npy', type=str)
parser.add_argument('--generated_low_root', default='./generated_imgs_low_level', type=str)
parser.add_argument('--generated_mix_root', default='./generated_imgs_low_and_high_level', type=str)
parser.add_argument('--dnn', default='clip_h14', type=str)
parser.add_argument('--num_inference_steps', default=10, type=int, help='num_inference_steps for diffusion model')
parser.add_argument('--img2img_strength', default=0.9, type=float, help='img2img_strength for low level image generation')

parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for model training')
parser.add_argument('--in_dim', default=1024, type=int, help='the dimension of input')
parser.add_argument('--num_tokens', default=1, type=int, help='the number of text tokens')
parser.add_argument('--clip_dim', default=1024, type=int, help='the dimension of clip text embeddings')
parser.add_argument('--n_blocks', default=2, type=int, help='the number of blocks in BrainNetwork')
parser.add_argument('--depth', default=2, type=int, help='the depth in PriorNetwork')
parser.add_argument('--num_sub', default=2,type=int, help='the number of subjects used in the experiments')
parser.add_argument('--batch_size', default=200, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
parser.add_argument('--val', default='GA_wo_baseline_corr_test', type=str)

parser.add_argument('--vae_infer_dtype', default='bfloat16', type=str,
                    help='dtype for vae_latent_projector inference: float32/float16/bfloat16')
parser.add_argument('--prior_embed_dtype', default='float16', type=str,
                    help='dtype for image_embeds passed into diffusion generator: float32/float16/bfloat16')
parser.add_argument('--latent_calibration', default='on', choices=['on', 'off'], type=str,
                    help='whether to align predicted latent stats to train VAE latent stats')
parser.add_argument('--precision_sweep', action='store_true',
                    help='run multiple precision settings sequentially for stability checks')
parser.add_argument('--sweep_dtypes', default='float32,bfloat16,float16', type=str,
                    help='comma-separated dtypes for --precision_sweep')
parser.add_argument('--max_samples', default=100, type=int,
                    help='number of test categories to generate per subject')
parser.add_argument('--num_variations', default=10, type=int,
                    help='number of images to generate for each category')
parser.add_argument('--subjects', default='sub-01,sub-02', type=str,
                    help='comma-separated subject ids to run, e.g. sub-01 or sub-01,sub-02')
parser.add_argument('--run_tag', default='', type=str,
                    help='optional suffix for output directory names to keep ablation runs separate')

class EEGToImageModule(nn.Module):
    def __init__(self):
        super(EEGToImageModule, self).__init__()
    def forward(self, x):
        return x

def load_category(args):
    data_path = args.test_index_path
    condi_path_test = pd.read_csv(data_path)
    condi_path_test = condi_path_test.iloc[:, 0]

    texts = []
    for condi_idx in range(args.max_samples):
        category_name = condi_path_test[condi_idx][1:-1].split('\\')[0]
        texts.append(category_name)
    return texts

def parse_torch_dtype(dtype_name: str):
    name = dtype_name.lower().strip()
    alias = {
        'fp32': 'float32',
        'fp16': 'float16',
        'bf16': 'bfloat16',
    }
    name = alias.get(name, name)
    if name == 'bfloat32':
        raise ValueError("`bfloat32` is not a valid torch dtype. Please use one of: float32, float16, bfloat16")
    mapping = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}. Use float32/float16/bfloat16")
    return mapping[name], name

def tensor_health(name, x):
    x32 = x.detach().to(torch.float32)
    has_nan = torch.isnan(x32).any().item()
    has_inf = torch.isinf(x32).any().item()
    print(
        f"[health] {name}: dtype={x.dtype}, shape={tuple(x.shape)}, "
        f"mean={x32.mean().item():.4f}, std={x32.std().item():.4f}, "
        f"min={x32.min().item():.4f}, max={x32.max().item():.4f}, nan={has_nan}, inf={has_inf}"
    )
    return (not has_nan) and (not has_inf)

def low_level_image_load(args, nSub):
    img_directory = os.path.join(args.generated_low_root, args.dnn, args.val, nSub)
    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()

    images = []
    for folder in all_folders:
        folder_path = os.path.join(img_directory, folder)
        image_names = sorted(os.listdir(folder_path))
        if not image_names:
            continue
        image_path = os.path.join(folder_path, image_names[0])
        print('The path of low_level images_recon:', image_path)
        test_low_level_image = Image.open(image_path).convert("RGB")
        images.append(test_low_level_image)
    return images

def main(): 
    args = parser.parse_args()
    if args.max_samples <= 0:
        raise ValueError('--max_samples must be > 0')
    if args.num_variations <= 0:
        raise ValueError('--num_variations must be > 0')
    vae_dtype, vae_dtype_name = parse_torch_dtype(args.vae_infer_dtype)
    prior_dtype, prior_dtype_name = parse_torch_dtype(args.prior_embed_dtype)

    if args.precision_sweep:
        sweep_dtype_names = [x.strip() for x in args.sweep_dtypes.split(',') if x.strip()]
    else:
        sweep_dtype_names = [vae_dtype_name]

    sweep_dtype_pairs = [parse_torch_dtype(x) for x in sweep_dtype_names]

    # Set seed value
    seed_n = args.seed
    print('seed is ' + str(seed_n))
    print(f"[config] prior_embed_dtype={prior_dtype_name}, latent_calibration={args.latent_calibration}, precision_sweep={args.precision_sweep}")
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_n)
    generator = Generator4Embeds_latent2img(num_inference_steps=args.num_inference_steps, img2img_strength=args.img2img_strength, device=device)
    subjects = [x.strip() for x in args.subjects.split(',') if x.strip()]
    run_tag = args.run_tag.strip()
    for sub in subjects:
        print('The subject is: ', sub)
        categories = load_category(args)
        print(categories)
        model_dir = os.path.join(args.models_root, args.dnn, args.val, sub)
        eeg_test_features = torch.load(os.path.join(model_dir, 'eeg_test_features.pt'), weights_only=True)['eeg_test_features']

        train_vae_latents = torch.load(os.path.join(args.vae_latent_root, 'train_vae_latents.pt'), weights_only=True)['train_vae_latents']
        test_idx_intrain = np.load(args.test_idx_path)
        train_vae_latents = np.delete(train_vae_latents.numpy(), test_idx_intrain, axis=0)
        train_vae_latents = torch.from_numpy(train_vae_latents)
        ref_mean = train_vae_latents.mean().to(device=device, dtype=torch.float32)
        ref_std = train_vae_latents.std().to(device=device, dtype=torch.float32)
        print(f"[latent ref][{sub}] mean/std: {ref_mean.item():.4f}/{ref_std.item():.4f}")
        
        print('The shape of eeg_test_features is:', eeg_test_features.shape)

        image_model = EEGToImageModule()
        # setup diffusion prior network
        timesteps = 100
        prior_network = PriorNetwork(
                dim=args.clip_dim,
                depth=args.depth,
                dim_head=256,
                heads=8,
                causal=False,
                num_tokens=args.num_tokens,
                learned_query_mode="pos_emb"
            ).to(device)

        image_model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=args.clip_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        ).to(device)

        image_model.load_state_dict(torch.load(os.path.join(model_dir, 'image_condition.pth'), weights_only=True), strict=False)
        image_model.eval()
        
        vae_ckpt = torch.load(os.path.join(model_dir, 'vae_condition.pth'), weights_only=True)

        for run_vae_dtype, run_vae_dtype_name in sweep_dtype_pairs:
            print(f"[precision-run] subject={sub}, vae_infer_dtype={run_vae_dtype_name}, prior_embed_dtype={prior_dtype_name}")
            vae_model = vae_latent_projector().to(run_vae_dtype).to(device)
            vae_model.load_state_dict(vae_ckpt, strict=False)
            vae_model.eval()

            if args.precision_sweep:
                run_dir = f"precision_{run_vae_dtype_name}"
            else:
                run_dir = ''

            if run_tag:
                run_dir = f"{run_dir}_{run_tag}" if run_dir else run_tag

            if run_dir:
                directory = os.path.join(args.generated_mix_root, args.dnn, 'wo_high_check', args.val, sub, run_dir)
            else:
                directory = os.path.join(args.generated_mix_root, args.dnn, 'wo_high_check', args.val, sub)

            with torch.no_grad():
                for k in range(args.max_samples):
                    eeg_img_embeds = eeg_test_features[k:k+1]
                    vae_latent_pred = vae_model(eeg_img_embeds.to(run_vae_dtype).to(device))

                    if args.latent_calibration == 'on':
                        pred_f32 = vae_latent_pred.to(torch.float32)
                        pred_mean = pred_f32.mean(dim=(1, 2, 3), keepdim=True)
                        pred_std = pred_f32.std(dim=(1, 2, 3), keepdim=True)
                        pred_f32 = (pred_f32 - pred_mean) / (pred_std + 1e-6)
                        pred_f32 = pred_f32 * ref_std + ref_mean
                        vae_latent_pred = pred_f32.to(run_vae_dtype)

                    prior_in = eeg_img_embeds.unsqueeze(0).to(device)
                    prior_out = image_model.diffusion_prior.p_sample_loop(prior_in.shape, text_cond=dict(text_embed=prior_in), cond_scale=1.0, timesteps=20)
                    image_embeds_pred = prior_out.squeeze(0)

                    if k == 0:
                        tensor_health(f"{sub}/{run_vae_dtype_name}/vae_latent_pred", vae_latent_pred)
                        tensor_health(f"{sub}/{run_vae_dtype_name}/image_embeds_pred", image_embeds_pred)

                    print('The shape of image_embeds_pred is:', image_embeds_pred.shape)
                    print('The shape of vae_latent_pred is:', vae_latent_pred.shape)

                    for j in range(args.num_variations):
                        image = generator.generate(image_embeds_pred.to(dtype=prior_dtype), vae_latent_pred, generator=gen)
                        path = f'{directory}/{categories[k]}/{j}.png'
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        image.save(path)
                        print(f'Image saved to {path}')

if __name__ == '__main__':
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
