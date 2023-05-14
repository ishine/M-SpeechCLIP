import argparse

def get_args(description='M-SpeechCLIP Hyperparameters and Training Arguments'):
    parser = argparse.ArgumentParser(description=description)
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=128, help='# samples per batch for training and evaluation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Peak learning rate to use during training')
    parser.add_argument('--schedule', type=str, default='lw10ld', help='LR scheduler to use. "None" indicates constant learning rate; "lw10ld" indicates linear warmup from 0 to the peak LR for the first 10%% of iterations, followed by linear decrease back to 0')
    parser.add_argument('--gpus', type=int, default=2, help='Number of gpus visible to cuda')
    parser.add_argument('--device', type=str, default='cuda', help='What type of device to use (cuda or cpu)')
    parser.add_argument('--logging', type=int, default=0, help='Whether to use tensorboard to log loss over the coarse of training')
    parser.add_argument('--log_dir', type=str, default='default_logs', help='Folder to dump tensorboard logs into')
    parser.add_argument('--epochs', type=int, default=5, help='How many epochs to train for; if 0, does testing only')
    parser.add_argument('--display', type=int, default=25, help='# iterations of training to wait between displaying the training loss')
    parser.add_argument('--workers', type=int, default=32, help='How many threads will be spawned by the data loader')
    parser.add_argument('--mono_batches', type=int, default=0, help='If 0, batches contain a random mix of languages; if 1, each batch is monolingual. Controls both training and testing setting')

    # Data and model paths
    parser.add_argument('--dataset', type=str, default='PlacesEng', help='Which dataset to train and test on (current options are Places (full 400k English), PlacesEng (100k subset English), PlacesHindi, PlacesJpn, and PlacesMulti)')
    parser.add_argument('--chkpt_path', type=str, default='/saltpool0/scratch/layneberry/CLIPC/default_checkpoint.pth', help='Where to save checkpoints (also used as the path to load from if load_chkpt is on')
    parser.add_argument('--load_chkpt', type=bool, default=False, help='Whether to load weights from a saved checkpoint (if on, location defaults to chkpt_path)')
    parser.add_argument('--load_from', type=str, default='', help='If specified, loads from a different location than ckpt_path')

    # Model architecture
    parser.add_argument('--model_type', type=str, default='Parallel', help='What type of differentiable layer to use to map from hubert_out to clip_in (current options are Parallel and LangID')
    parser.add_argument('--hubert_size', type=str, default='large', help='Use base or large variant of HuBERT?')
    parser.add_argument('--clip_size', type=str, default='large', help='Use base or large variant of CLIP?')
    parser.add_argument('--heads', type=int, default=8, help='How many attention heads to use for SpeechCLIP-P')
    parser.add_argument('--layers', type=int, default=1, help='How many transformer layers to use for SpeechCLIP-P')
    parser.add_argument('--feat_trainable', type=int, default=0, help='Whether the feature extractor should be trainable (1) or frozen (0)')
    parser.add_argument('--weighted_sum', type=int, default=1, help='Whether to use a weighted sum (1) over feature extractor layers or extract the final layer (0)')

    # Loss Specification
    parser.add_argument('--loss_type', type=str, default='MMS', help='Type of loss to use (current options are MMS and CrossLingual)')
    parser.add_argument('--cross_scale', type=float, default=0.01, help='Factor to scale cross-lingual retrieval loss terms by, if loss type is CrossLingual')
    parser.add_argument('--use_all_three', type=int, default=1, help='When including cross-lingual loss terms, whether to use all 3 languages at each epoch or sample 2')
    
    args = parser.parse_args()
    return args
