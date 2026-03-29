"""
Comprehensive reconstruction comparison with multiple metrics, 
outputting CSV of mean/std and SVG of best/median/worst examples for each metric.

The final used one
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import scipy as sp
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec

from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import clip


def is_image_file(name):
    lower = name.lower()
    return lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg') or lower.endswith('.bmp')


def natural_image_sort_key(name):
    stem = os.path.splitext(name)[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, device, feature_layer=None, return_avg=True):
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))

    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images) - 1)
        return perf
    return success_cnt, len(all_images) - 1


@torch.no_grad()
def com_corrcoef(brain_recon, img, model, preprocess, device, feature_layer=None):
    preds = model(preprocess(brain_recon).to(device))
    reals = model(preprocess(img).to(device))

    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    return float(np.corrcoef(reals, preds)[0][1])


def pick_indices(values, higher_better):
    arr = np.array(values, dtype=np.float64)
    order = np.argsort(arr)
    if higher_better:
        order = order[::-1]
    n = len(order)
    best = [int(order[0]), int(order[1 if n > 1 else 0])]
    med = [int(order[max(0, n // 2 - 1)]), int(order[min(n - 1, n // 2)])]
    worst = [int(order[-2 if n > 1 else -1]), int(order[-1])]
    return best + med + worst


def main():
    parser = argparse.ArgumentParser(description='Reconstruction comparison metrics with mean/std')
    parser.add_argument('--THINGS_dir', default='./data/THINGS/Images', type=str)
    parser.add_argument('--test_index_path', default='./TVSD/things_imgs_test.csv', type=str)
    parser.add_argument('--generated_root', default='./generated_imgs_low_and_high_level', type=str)
    parser.add_argument('--results_root', default='./results', type=str)
    parser.add_argument('--dnn', default='clip_h14', type=str)
    parser.add_argument('--val', default='wo_high_check', type=str)
    parser.add_argument('--sub', default='sub-02', type=str)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--viz_resize', default=256, type=int)
    args = parser.parse_args()

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    set_seed(args.seed)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['svg.fonttype'] = 'none'

    source_dir = f"{args.generated_root}/{args.dnn}/{args.val}/{args.sub}"
    out_dir = os.path.join(args.results_root, args.dnn, args.val)
    os.makedirs(out_dir, exist_ok=True)

    test_series = pd.read_csv(args.test_index_path).iloc[:, 0]
    categories = [test_series[i][1:-1].split('\\')[0] for i in range(len(test_series))]

    # load generated ordered by CSV category
    generated = []
    for cat in categories:
        folder = os.path.join(source_dir, cat)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f'Missing generated folder: {folder}')
        files = sorted([f for f in os.listdir(folder) if is_image_file(f)], key=natural_image_sort_key)
        images = []
        for f in files:
            with Image.open(os.path.join(folder, f)) as img:
                images.append(torch.from_numpy(np.array(img.convert('RGB'))))
        generated.append(torch.stack(images, dim=0))

    generated = torch.stack(generated, dim=0)   # [N, K, H, W, C]
    N, K, H, W, _ = generated.shape
    print('Generated shape:', generated.shape)

    # load real and resize to generated size
    reals = []
    for i in range(len(test_series)):
        rel = test_series[i].replace('\\', '/')[1:-1]
        p = os.path.join(args.THINGS_dir, rel)
        with Image.open(p) as img:
            img = img.convert('RGB').resize((W, H), Image.BILINEAR)
            reals.append(torch.from_numpy(np.array(img)))
    reals = torch.stack(reals, dim=0)  # [N,H,W,C]
    print('Real shape:', reals.shape)

    real_uint8 = reals.numpy()
    gen_uint8 = generated.numpy()

    all_images = reals.permute(0, 3, 1, 2).to(device).float()
    all_brain_recons = generated.permute(0, 1, 4, 2, 3).reshape(N * K, 3, H, W).to(device).float()

    vis_real = transforms.Resize((args.viz_resize, args.viz_resize))(all_images)
    vis_recon = transforms.Resize((args.viz_resize, args.viz_resize))(all_brain_recons)

    # ---------------- PixCorr ----------------
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Resize((425, 425), interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    all_images_flat = preprocess(vis_real).reshape(len(vis_real), -1).cpu().numpy()
    all_recons_flat = preprocess(vis_recon).reshape(len(vis_recon), -1).cpu().numpy()

    pixcorr_per_cat = []
    pixcorr_mat = np.zeros((N, K), dtype=np.float64)
    for i in tqdm(range(N), desc='PixCorr'):
        corr_list = []
        for j in range(K):
            idx = i * K + j
            corr = np.corrcoef(all_images_flat[i], all_recons_flat[idx])[0][1]
            corr_list.append(corr)
            pixcorr_mat[i, j] = corr
        pixcorr_per_cat.append(float(np.mean(corr_list)))
    pixcorr_mean = float(np.mean(pixcorr_per_cat))
    pixcorr_std = float(np.std(pixcorr_per_cat))

    # ---------------- SSIM ----------------
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    img_gray = rgb2gray(preprocess(vis_real).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess(vis_recon).permute((0, 2, 3, 1)).cpu())

    ssim_per_cat = []
    ssim_mat = np.zeros((N, K), dtype=np.float64)
    for i in tqdm(range(N), desc='SSIM'):
        ssim_list = []
        for j in range(K):
            idx = i * K + j
            s = ssim(recon_gray[idx], img_gray[i], gaussian_weights=True, sigma=1.5,
                     use_sample_covariance=False, data_range=1.0)
            ssim_list.append(s)
            ssim_mat[i, j] = s
        ssim_per_cat.append(float(np.mean(ssim_list)))
    ssim_mean = float(np.mean(ssim_per_cat))
    ssim_std = float(np.std(ssim_per_cat))

    # ---------------- AlexNet ----------------
    alex_model = create_feature_extractor(alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), return_nodes=['features.4', 'features.11']).to(device)
    alex_model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.float() / 255),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    recons_alex2 = torch.zeros_like(vis_recon)
    recons_alex5 = torch.zeros_like(vis_recon)
    alex2_per_cat = []
    alex5_per_cat = []
    for i in tqdm(range(N), desc='AlexNet sim'):
        original = vis_real[i].unsqueeze(0)
        sims2, sims5 = [], []
        for j in range(K):
            recon = vis_recon[i * K + j].unsqueeze(0)
            sims2.append(com_corrcoef(recon, original, alex_model, preprocess, device, 'features.4'))
            sims5.append(com_corrcoef(recon, original, alex_model, preprocess, device, 'features.11'))
        alex2_per_cat.append(float(np.mean(sims2)))
        alex5_per_cat.append(float(np.mean(sims5)))

        s2 = sorted(sims2)
        s5 = sorted(sims5)
        for k in range(K):
            idx2 = sims2.index(s2[k])
            idx5 = sims5.index(s5[k])
            recons_alex2[i * K + k] = vis_recon[i * K + idx2]
            recons_alex5[i * K + k] = vis_recon[i * K + idx5]

    alex2s, alex5s = [], []
    for i in range(K):
        indices = torch.arange(i, len(vis_recon), step=K)
        alex2s.append(two_way_identification(recons_alex2[indices], vis_real, alex_model, preprocess, device, 'features.4'))
        alex5s.append(two_way_identification(recons_alex5[indices], vis_real, alex_model, preprocess, device, 'features.11'))
    alex2_mean, alex2_std = float(np.mean(alex2s)), float(np.std(alex2s))
    alex5_mean, alex5_std = float(np.mean(alex5s)), float(np.std(alex5s))

    # ---------------- Inception ----------------
    inception_model = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    recons_incep = torch.zeros_like(vis_recon)
    incep_per_cat = []
    for i in tqdm(range(N), desc='Inception sim'):
        original = vis_real[i].unsqueeze(0)
        sims = []
        for j in range(K):
            recon = vis_recon[i * K + j].unsqueeze(0)
            sims.append(com_corrcoef(recon, original, inception_model, preprocess, device, 'avgpool'))
        incep_per_cat.append(float(np.mean(sims)))
        s_sorted = sorted(sims)
        for k in range(K):
            idx = sims.index(s_sorted[k])
            recons_incep[i * K + k] = vis_recon[i * K + idx]

    inceptions = []
    for i in range(K):
        indices = torch.arange(i, len(vis_recon), step=K)
        inceptions.append(two_way_identification(recons_incep[indices], vis_real, inception_model, preprocess, device, 'avgpool'))
    incep_mean, incep_std = float(np.mean(inceptions)), float(np.std(inceptions))

    # ---------------- CLIP ----------------
    clip_model, _ = clip.load('ViT-L/14', device=device)
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    recons_clip = torch.zeros_like(vis_recon)
    clip_per_cat = []
    for i in tqdm(range(N), desc='CLIP sim'):
        original = vis_real[i].unsqueeze(0)
        sims = []
        for j in range(K):
            recon = vis_recon[i * K + j].unsqueeze(0)
            sims.append(com_corrcoef(recon, original, clip_model.encode_image, preprocess, device, None))
        clip_per_cat.append(float(np.mean(sims)))
        s_sorted = sorted(sims)
        for k in range(K):
            idx = sims.index(s_sorted[k])
            recons_clip[i * K + k] = vis_recon[i * K + idx]

    clips = []
    for i in range(K):
        indices = torch.arange(i, len(vis_recon), step=K)
        clips.append(two_way_identification(recons_clip[indices], vis_real, clip_model.encode_image, preprocess, device, None))
    clip_mean, clip_std = float(np.mean(clips)), float(np.std(clips))

    # ---------------- EfficientNet ----------------
    eff_model = create_feature_extractor(efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), return_nodes=['avgpool']).to(device)
    eff_model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gt = eff_model(preprocess(vis_real))['avgpool'].reshape(len(vis_real), -1).cpu().numpy()
    fake = []
    for i in range(0, len(vis_recon), 100):
        batch = vis_recon[i:i + 100]
        fake.append(eff_model(preprocess(batch))['avgpool'].reshape(len(batch), -1).cpu().numpy())
    fake = np.concatenate(fake, axis=0)

    eff_per_cat = []
    for i in tqdm(range(N), desc='EfficientNet distance'):
        ds = []
        for j in range(K):
            ds.append(sp.spatial.distance.correlation(gt[i], fake[i * K + j]))
        eff_per_cat.append(float(np.mean(ds)))
    eff_mean, eff_std = float(np.mean(eff_per_cat)), float(np.std(eff_per_cat))

    # ---------------- SwAV ----------------
    swav_backbone = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(swav_backbone, return_nodes=['avgpool']).to(device)
    swav_model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gt = swav_model(preprocess(vis_real))['avgpool'].reshape(len(vis_real), -1).cpu().numpy()
    fake = []
    for i in range(0, len(vis_recon), 100):
        batch = vis_recon[i:i + 100]
        fake.append(swav_model(preprocess(batch))['avgpool'].reshape(len(batch), -1).cpu().numpy())
    fake = np.concatenate(fake, axis=0)

    swav_per_cat = []
    for i in tqdm(range(N), desc='SwAV distance'):
        ds = []
        for j in range(K):
            ds.append(sp.spatial.distance.correlation(fake[i * K + j], gt[i]))
        swav_per_cat.append(float(np.mean(ds)))
    swav_mean, swav_std = float(np.mean(swav_per_cat)), float(np.std(swav_per_cat))

    # ---- CSV only mean/std ----
    df = pd.DataFrame({
        'Metric': ['PixCorr', 'SSIM', 'AlexNet(2)', 'AlexNet(5)', 'Inception', 'CLIP', 'EfficientNet', 'SwAV'],
        'mean': [pixcorr_mean, ssim_mean, alex2_mean, alex5_mean, incep_mean, clip_mean, eff_mean, swav_mean],
        'std': [pixcorr_std, ssim_std, alex2_std, alex5_std, incep_std, clip_std, eff_std, swav_std],
    })
    csv_path = f"{out_dir}/recon_metrics_low_mean_std_{args.sub}.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))

    metric_panels = [
        ('PixCorr', pixcorr_per_cat, True),
        ('SSIM', ssim_per_cat, True),
        ('AlexNet(2)', alex2_per_cat, True),
        ('AlexNet(5)', alex5_per_cat, True),
        ('Inception', incep_per_cat, True),
        ('CLIP', clip_per_cat, True),
        ('EfficientNet', eff_per_cat, False),
        ('SwAV', swav_per_cat, False),
    ]

    fig = plt.figure(figsize=(30, 16))
    outer = gridspec.GridSpec(2, 4, figure=fig, wspace=0.25, hspace=0.35)
    for m_idx, (name, per_cat, higher_better) in enumerate(metric_panels):
        row, col = divmod(m_idx, 4)
        subgrid = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=outer[row, col], wspace=0.02, hspace=0.02)

        selected = pick_indices(per_cat, higher_better)
        rep_recon_idx = np.argmax(pixcorr_mat, axis=1)

        for c in range(6):
            cat_idx = selected[c]
            ridx = int(rep_recon_idx[cat_idx])
            tag = ['B1', 'B2', 'M1', 'M2', 'W1', 'W2'][c]

            ax1 = fig.add_subplot(subgrid[0, c])
            ax1.imshow(real_uint8[cat_idx])
            ax1.axis('off')
            if c == 0:
                ax1.set_ylabel('GT', fontsize=8)
            ax1.set_title(f'{tag}\\n {categories[cat_idx]}', fontsize=7)

            ax2 = fig.add_subplot(subgrid[1, c])
            ax2.imshow(gen_uint8[cat_idx, ridx])
            ax2.axis('off')
            if c == 0:
                ax2.set_ylabel('Recon', fontsize=8)

        t = fig.add_subplot(outer[row, col])
        t.set_xticks([])
        t.set_yticks([])
        t.patch.set_alpha(0.0)
        for spine in t.spines.values():
            spine.set_visible(False)
        t.set_title(f'{name} (mean={np.mean(per_cat):.4f}, std={np.std(per_cat):.4f})', fontsize=11, pad=8)

    fig.suptitle(f'Reconstruction Comparison ({args.dnn} | {args.val} | {args.sub})', fontsize=14)
    svg_path = f"{out_dir}/reconstruction_low_comparison_{args.sub}.svg"
    fig.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved CSV: {csv_path}')
    print(f'Saved SVG: {svg_path}')


if __name__ == '__main__':
    main()
