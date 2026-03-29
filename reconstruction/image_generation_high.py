import os
import torch
import time
import numpy as np
import torch.nn as nn
import pandas as pd
import random
import argparse
try:
    from .custom_pipeline_high import Generator4Embeds
    from .model import PriorNetwork, BrainDiffusionPrior
except ImportError:
    from custom_pipeline_high import Generator4Embeds
    from model import PriorNetwork, BrainDiffusionPrior

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Visual stimuli reconstruction with EEG")
parser.add_argument('--test_index_path', default='./TVSD/things_imgs_test.csv', type=str)
parser.add_argument('--models_root', default='./models', type=str)
parser.add_argument('--generated_root', default='./generated_imgs_high_level', type=str)
parser.add_argument('--dnn', default='clip_h14', type=str)

parser.add_argument('--lr', default=3e-4, type=float, help='learning rate for model training')
parser.add_argument('--in_dim', default=1024, type=int, help='the dimension of input')
parser.add_argument('--num_tokens', default=1, type=int, help='the number of text tokens')
parser.add_argument('--clip_dim', default=1024, type=int, help='the dimension of clip text embeddings')
parser.add_argument('--n_blocks', default=2, type=int, help='the number of blocks in BrainNetwork')
parser.add_argument('--depth', default=2, type=int, help='the depth in PriorNetwork')
parser.add_argument('--num_sub', default=2,type=int, help='the number of subjects used in the experiments')
parser.add_argument('--batch_size', default=200, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training')
parser.add_argument('--val', default='GA_wo_baseline_corr', type=str)

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
    for condi_idx in range(100):
        category_name = condi_path_test[condi_idx][1:-1].split('\\')[0]
        texts.append(category_name)
    return texts


def main(): 
    args = parser.parse_args()
    # Set seed value
    seed_n = args.seed
    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_n)
    
    generator = Generator4Embeds(num_inference_steps=4, device=device)
    subjects = ['sub-01', 'sub-02']
    for sub in subjects:
        print('The subject is: ', sub)
        categories = load_category(args)
        print(categories)
        model_dir = os.path.join(args.models_root, args.dnn, args.val, sub)
        eeg_test_features = torch.load(os.path.join(model_dir, 'eeg_test_features.pt'), weights_only=True)['eeg_test_features']
        
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
        
        directory = os.path.join(args.generated_root, args.dnn, args.val, sub)
        with torch.no_grad():
            for k in range(100):
                eeg_img_embeds = eeg_test_features[k:k+1]
                
                prior_in = eeg_img_embeds.unsqueeze(0).to(device)
                prior_out = image_model.diffusion_prior.p_sample_loop(prior_in.shape, text_cond=dict(text_embed=prior_in), cond_scale=1.0, timesteps=20)
                image_embeds_pred = prior_out.squeeze(0)
                print('The shape of image_embeds_pred is:', image_embeds_pred.shape)

                for j in range(10):
                    image = generator.generate(image_embeds_pred.to(dtype=torch.float16), generator=gen)
                    path = f'{directory}/{categories[k]}/{j}.png'
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    # Save the PIL Image
                    image.save(path)
                    print(f'Image saved to {path}')

if __name__ == '__main__':
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))

