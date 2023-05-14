# Constants/hyperparameters
from args import get_args
args = get_args()

# Declaring all the paths up top
# If not pre-dumping image embeddings, comment out the line below and remove image_path from model arguments
image_path = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/image_encodings_from_clip_large.pkl"
if args.dataset == 'Places':
    train_data_path = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/PlacesEnglishFull_train.json"
    val_data_path = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/PlacesEnglishFull_val.json"
    test_data_path = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/PlacesEnglishFull_test.json"
elif args.dataset == 'PlacesEng':
    train_data_path = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/PlacesEnglishSubset_train.json"
    val_data_path = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/PlacesEnglishSubset_val.json"
    test_data_path = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/PlacesEnglishSubset_test.json"
elif args.dataset == 'PlacesHindi':
    train_data_path = "/saltpool0/data/layneberry/hindi_places_100k/PlacesHindi_train.json"
    val_data_path = "/saltpool0/data/layneberry/hindi_places_100k/PlacesHindi_val.json"
    test_data_path = "/saltpool0/data/layneberry/hindi_places_100k/PlacesHindi_test.json"
elif args.dataset == 'PlacesJpn':
    train_data_path = "/saltpool0/data/layneberry/PlacesAudioJpn_100k/PlacesJpn_train.json"
    val_data_path = "/saltpool0/data/layneberry/PlacesAudioJpn_100k/PlacesJpn_val.json"
    test_data_path = "/saltpool0/data/layneberry/PlacesAudioJpn_100k/PlacesJpn_test.json"
elif args.dataset == 'PlacesMulti':
    train_eng = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/PlacesEnglishSubset_train.json"
    val_eng = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/PlacesEnglishSubset_val.json"
    test_eng = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/PlacesEnglishSubset_test.json"
    train_jpn = "/saltpool0/data/layneberry/PlacesAudioJpn_100k/PlacesJpn_train.json"
    val_jpn = "/saltpool0/data/layneberry/PlacesAudioJpn_100k/PlacesJpn_val.json"
    test_jpn = "/saltpool0/data/layneberry/PlacesAudioJpn_100k/PlacesJpn_test.json"
    train_hindi = "/saltpool0/data/layneberry/hindi_places_100k/PlacesHindi_train.json"
    val_hindi = "/saltpool0/data/layneberry/hindi_places_100k/PlacesHindi_val.json"
    test_hindi = "/saltpool0/data/layneberry/hindi_places_100k/PlacesHindi_test.json"

# Import everything 
from metrics import evaluate
from train_step import train_step
from torch.utils.data import DataLoader
from places_dataloader import Places_DataLoader, Multilingual_Places_DataLoader, Multilingual_Places_DataLoader_All
from loss import MMS_loss
from tqdm import tqdm
import clip
import pickle as pkl
import time, random
import torch as th
from args import get_args
from torch.cuda.amp import autocast, GradScaler

args = get_args()

# Instantiate the model and load weights
# If not pre-dumping image embeddings, uncomment the next line and add clip_preprocess to model arguments
# _, clip_preprocess = clip.load('ViT-B/32',device='cpu')

if args.model_type == 'Parallel':
    from parallel_model import Parallel
    model = Parallel(heads=args.heads, layers=args.layers, batch=args.batch_size, gpus=args.gpus, feat_trainable=args.feat_trainable, weighted_sum=args.weighted_sum, hubert_size=args.hubert_size, clip_size=args.clip_size)
    if args.device == 'cuda' and args.gpus > 1:
        model = th.nn.DataParallel(model)
elif args.model_type == 'LangID':
    from parallel_model import Parallel
    model = Parallel(heads=args.heads, layers=args.layers, batch=args.batch_size, gpus=args.gpus, feat_trainable=args.feat_trainable, weighted_sum=args.weighted_sum, hubert_size=args.hubert_size, use_langID=True)
    if args.device == 'cuda' and args.gpus > 1:
        model = th.nn.DataParallel(model)
else:
    raise Exception('Unrecognized model type')

if args.load_chkpt:
    if args.load_from != '':
        model.load_state_dict(th.load(args.load_from), strict=False)
    else:
        model.load_state_dict(th.load(args.chkpt_path), strict=False)

model = model.to(args.device)
   
# Instantiate each dataset (train, val, test)
if args.epochs > 0:
    if args.dataset in ['Places', 'PlacesEng']:
        train_dataset = DataLoader(Places_DataLoader(train_data_path, image_path, language='English'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    elif args.dataset == 'PlacesHindi':
        train_dataset = DataLoader(Places_DataLoader(train_data_path, image_path, language='Hindi'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    elif args.dataset == 'PlacesJpn':
        train_dataset = DataLoader(Places_DataLoader(train_data_path, image_path, language='Japanese'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    elif args.dataset == 'PlacesMulti' and args.mono_batches:
        train_dataset_eng = DataLoader(Places_DataLoader(train_eng, image_path, language='English'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
        train_dataset_hindi = DataLoader(Places_DataLoader(train_hindi, image_path, language='Hindi'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
        train_dataset_jpn = DataLoader(Places_DataLoader(train_jpn, image_path, language='Japanese'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    elif args.dataset == 'PlacesMulti' and args.loss_type != 'CrossLingual':
        train_dataset = DataLoader(Multilingual_Places_DataLoader(train_eng, train_jpn, train_hindi, image_path),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    elif args.dataset == 'PlacesMulti':
        train_dataset = DataLoader(Multilingual_Places_DataLoader_All(train_eng, train_jpn, train_hindi, image_path),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    
if args.dataset in ['Places', 'PlacesEng']:
    val_dataset = DataLoader(Places_DataLoader(val_data_path, image_path, language='English'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
elif args.dataset == 'PlacesHindi':
    val_dataset = DataLoader(Places_DataLoader(val_data_path, image_path, language='Hindi'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
elif args.dataset == 'PlacesJpn':
    val_dataset = DataLoader(Places_DataLoader(val_data_path, image_path, language='Japanese'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
elif args.dataset == 'PlacesMulti' and args.mono_batches:
    val_dataset_eng = DataLoader(Places_DataLoader(val_eng, image_path, language='English'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    val_dataset_hindi = DataLoader(Places_DataLoader(val_hindi, image_path, language='Hindi'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    val_dataset_jpn = DataLoader(Places_DataLoader(val_jpn, image_path, language='Japanese'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
elif args.dataset == 'PlacesMulti' and args.loss_type != 'CrossLingual':
    val_dataset = DataLoader(Multilingual_Places_DataLoader(val_eng, val_jpn, val_hindi, image_path),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
elif args.dataset == 'PlacesMulti':
    val_dataset = DataLoader(Multilingual_Places_DataLoader_All(val_eng, val_jpn, val_hindi, image_path),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)

if args.dataset in ['Places', 'PlacesEng']:
    test_dataset = DataLoader(Places_DataLoader(test_data_path, image_path, language='English'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
elif args.dataset == 'PlacesHindi':
    test_dataset = DataLoader(Places_DataLoader(test_data_path, image_path, language='Hindi'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
elif args.dataset == 'PlacesJpn':
    test_dataset = DataLoader(Places_DataLoader(test_data_path, image_path, language='Japanese'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
elif args.dataset == 'PlacesMulti' and args.mono_batches:
    test_dataset_eng = DataLoader(Places_DataLoader(test_eng, image_path, language='English'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    test_dataset_hindi = DataLoader(Places_DataLoader(test_hindi, image_path, language='Hindi'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    test_dataset_jpn = DataLoader(Places_DataLoader(test_jpn, image_path, language='Japanese'),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
elif args.dataset == 'PlacesMulti' and args.loss_type != 'CrossLingual':
    test_dataset = DataLoader(Multilingual_Places_DataLoader(test_eng, test_jpn, test_hindi, image_path),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
elif args.dataset == 'PlacesMulti':
    test_dataset = DataLoader(Multilingual_Places_DataLoader_All(test_eng, test_jpn, test_hindi, image_path),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True)

# Instantiate loss function, optimizer, etc for training
scaler = GradScaler()
loss_op = MMS_loss()
opt = th.optim.Adam(model.parameters(), lr=args.lr)

if args.schedule == 'lw10ld' and args.epochs > 0:
    # lwd10 = "Linear Warm-Up for 10% of iterations, then Linear Decay
    from torch.optim.lr_scheduler import SequentialLR, LinearLR
    try:
        total = args.epochs * (len(train_dataset))
    except:
        total = args.epochs * (len(train_dataset_eng))
    inflect = total // 10
    print('LR scheduler found total iterations', total, 'and inflection point', inflect)
    warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=inflect)
    decay = LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=total-inflect)
    scheduler = SequentialLR(opt, schedulers=[warmup, decay], milestones=[inflect])

# Training loop
itr = 1
running_total_loss = 0.0
running_eng = 0.0
running_jpn = 0.0
running_hindi = 0.0
running_ei = 0.0
running_ji = 0.0
running_hi = 0.0
running_ej = 0.0
running_eh = 0.0
running_jh = 0.0
running_seen = [0, 0, 0, 0, 0, 0] # ei, ji, hi, ej, eh, jh 
for e in range(args.epochs):
    print('Epoch', e)
    model.train()
    if not args.mono_batches:
        for batch in tqdm(train_dataset):
            model.zero_grad()
        
            with th.set_grad_enabled(True):
                with autocast():
                    if args.loss_type == 'CrossLingual' and not args.use_all_three:
                        lang_to_skip = random.choice(['Eng','Jpn','Hindi'])
                    else:
                        lang_to_skip=None
                    losses = train_step(args, batch, model, loss_op, lang_to_skip)
                    try:
                        total_loss = losses[0]
                    except:
                        total_loss = losses

                    if args.loss_type == 'MMS':
                        running_total_loss += losses.item()
                    elif args.loss_type == 'CrossLingual':
                        if args.use_all_three:
                            running_total_loss += losses[0].item()
                            running_ei += losses[1].item()
                            running_ji += losses[2].item()
                            running_hi += losses[3].item()
                            running_ej += losses[4].item()
                            running_eh += losses[5].item()
                            running_jh += losses[6].item()
                            for rs in range(len(running_seen)): running_seen[rs] += 1
                        elif lang_to_skip == 'Eng':
                            running_total_loss += losses[0].item()
                            running_ji += losses[1].item()
                            running_hi += losses[2].item()
                            running_jh += losses[3].item()
                            running_seen[1] += 1
                            running_seen[2] += 1
                            running_seen[5] += 1
                        elif lang_to_skip == 'Jpn':
                            running_total_loss += losses[0].item()
                            running_ei += losses[1].item()
                            running_hi += losses[2].item()
                            running_eh += losses[3].item()
                            running_seen[0] += 1
                            running_seen[2] += 1
                            running_seen[4] += 1
                        elif lang_to_skip == 'Hindi':
                            running_total_loss += losses[0].item()
                            running_ei += losses[1].item()
                            running_ji += losses[2].item()
                            running_ej += losses[3].item()
                            running_seen[0] += 1
                            running_seen[1] += 1
                            running_seen[3] += 1
                # Backprop and step
                scaler.scale(total_loss).backward()
                scaler.step(opt)
                scaler.update()

                if args.schedule != 'None':
                    scheduler.step()
            
                # If at display interval, print loss
                if itr % args.display == 0:
                    print("Training loss at iteration", itr, running_total_loss / args.display)
                    running_total_loss = 0.0
                    if args.loss_type == 'CrossLingual':
                        print('\tE2I:', running_ei / running_seen[0])
                        print('\tJ2I:', running_ji / running_seen[1])
                        print('\tH2I:', running_hi / running_seen[2])
                        print('\tE2J:', running_ej / running_seen[3])
                        print('\tE2H:', running_eh / running_seen[4])
                        print('\tJ2H:', running_jh / running_seen[5])
                        running_ei = 0.0
                        running_ji = 0.0
                        running_hi = 0.0
                        running_ej = 0.0
                        running_eh = 0.0
                        running_jh = 0.0
                        running_seen = [0 for _ in range(6)]
                itr += 1
                loop_end = time.time()
    else:
        # Three dataloaders, need to loop together
        # To do that, use iter
        assert(args.loss_type == 'MMS') # XLL w/ mono batches makes no sense
        train_hindi_iter = iter(train_dataset_hindi)
        train_jpn_iter = iter(train_dataset_jpn)
        for eng_batch in tqdm(train_dataset_eng):
            hindi_batch = next(train_hindi_iter)
            jpn_batch = next(train_jpn_iter)
            
            model.zero_grad()
            with th.set_grad_enabled(True):
                with autocast():
                    loss_eng = train_step(args, eng_batch, model, loss_op)
                    running_eng += loss_eng.item()
                scaler.scale(loss_eng).backward()
                with autocast():
                    loss_jpn = train_step(args, jpn_batch, model, loss_op)
                    running_jpn += loss_jpn.item()
                scaler.scale(loss_jpn).backward()
                with autocast():
                    loss_hindi = train_step(args, hindi_batch, model, loss_op)
                    running_hindi += loss_hindi.item()
                scaler.scale(loss_hindi).backward()
                scaler.step(opt)
                scaler.update()

                running_total_loss += loss_eng.item() + loss_jpn.item() + loss_hindi.item()

                if args.schedule != 'None':
                    scheduler.step()
                
                # If at display interval, print loss
                if itr % args.display == 0:
                    print("Total loss at iteration", itr, running_total_loss / args.display)
                    running_total_loss = 0.0
                    print('\tEnglish Loss:', running_eng / args.display)
                    print('\tJapanese Loss:', running_jpn / args.display)
                    print('\tHindi Loss:', running_hindi / args.display)
                    running_eng = 0.0
                    running_jpn = 0.0
                    running_hindi = 0.0
                itr += 1

    # In outer loop, report metrics on val set and save model
    if args.dataset == 'PlacesMulti' and args.mono_batches:
        print('English Val:')
        evaluate(model, val_dataset_eng, clip_size=args.clip_size)
        print('Japanese Val:')
        evaluate(model, val_dataset_jpn, clip_size=args.clip_size)
        print('Hindi Val:')
        evaluate(model, val_dataset_hindi, clip_size=args.clip_size)
    else:
        print('Mixed Val:')
        evaluate(model, val_dataset, clip_size=args.clip_size, loss_type=args.loss_type)
    
    if args.chkpt_path:
        th.save(model.state_dict(), args.chkpt_path)
        if args.loss_type == 'GAN':
            th.save(discrim.state_dict(), args.chkpt_path[:-4]+'_discriminator.pth')

# Full evaluation on both val and test sets
print('Val Set results:')
if args.dataset == 'PlacesMulti' and args.mono_batches:
    print('English Val:')
    evaluate(model, val_dataset_eng, clip_size=args.clip_size)
    print('Japanese Val:')
    evaluate(model, val_dataset_jpn, clip_size=args.clip_size)
    print('Hindi Val:')
    evaluate(model, val_dataset_hindi, clip_size=args.clip_size) 
else:
    print('Mixed Val:')
    evaluate(model, val_dataset, clip_size=args.clip_size, loss_type=args.loss_type)
print('Test Set results:')
if args.dataset == 'PlacesMulti' and args.mono_batches:
    print('English Test:')
    evaluate(model, val_dataset_eng, clip_size=args.clip_size)
    print('Japanese Test:')
    evaluate(model, val_dataset_jpn, clip_size=args.clip_size)
    print('Hindi Test:')
    evaluate(model, val_dataset_hindi, clip_size=args.clip_size) 
else:
    print('Mixed Test:')
    evaluate(model, val_dataset, clip_size=args.clip_size, loss_type=args.loss_type)

# Save model weights again for safe-keeping
if args.chkpt_path and args.epochs != 0:
    th.save(model.state_dict(), args.chkpt_path)
