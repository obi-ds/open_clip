import argparse
import torch
import matplotlib.pyplot as plt
from open_clip import create_model_and_transforms
from training.data import get_data
from training.params import parse_args

def parse_args_for_reconstruction():
    parser = argparse.ArgumentParser(description='Plot MAE reconstructions')
    parser.add_argument('--dataset-type', type=str, required=True, help='Dataset type')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--val-data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--val-num-samples', type=int, default=5, help='Number of validation samples')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to plot')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--patch-size', type=int, default=20, help='Patch size for tokenization')
    
    return parser.parse_args()

def main():
    args = parse_args_for_reconstruction()
    #args['train_data'] = None
    args.train_data = None
    args.train_num_samples = 0

    # Load the trained model
    model, _, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision='fp32',
        #device='cuda' if torch.cuda.is_available() else 'cpu',
        device='cpu',
        output_dict=True,
    )
    model.eval()

    # Prepare the data
    data = get_data(args, (None, preprocess_val), epoch=0, tokenizer=None)
    val_dataloader = data['val'].dataloader

    # Plot reconstructions
    for i, (images, _) in enumerate(val_dataloader):
        if i >= args.num_samples:
            break

        with torch.no_grad():

            outputs = model(images)
            predictions = outputs['predictions']
            reconstructions = outputs['reconstructions']
            labels = outputs['labels']
            mask = outputs['mask']

        # Plot original and reconstructed 12-lead ECGs
        fig, axes = plt.subplots(6, 2, figsize=(20, 30))
        fig.suptitle('Original (Blue) vs Reconstructed (Red) 12-Lead ECG', fontsize=16)

        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        for idx, ax in enumerate(axes.flatten()):
            # Original ECG in blue
            ax.plot(images[0, idx].numpy(), color='blue', label='Original')
            # Reconstructed ECG in red
            ax.plot(reconstructions[0, idx].numpy(), color='red', label='Reconstructed')

            # Plot mask
            mask_values = mask[0].repeat_interleave(args.patch_size).numpy()
            ax.fill_between(range(len(mask_values)), 0, 1, where=mask_values==1, 
                            color='gray', alpha=0.3, transform=ax.get_xaxis_transform(), 
                            label='Masked')
            
            ax.set_title(f'Lead {lead_names[idx]}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Voltage')
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'reconstruction_sample_{i}.png')
        plt.close()

if __name__ == '__main__':
    main()
