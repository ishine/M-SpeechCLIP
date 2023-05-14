from metrics import compute_metrics
import torch as th
from torch.utils.data import DataLoader
from places_dataloader import Places_DataLoader, Multilingual_Places_DataLoader, Multilingual_Places_DataLoader_All
import clip
from parallel_model import Parallel
from tqdm import tqdm

clip_model, img_preprocess = clip.load('ViT-L/14',device='cuda')
clip_model.eval()

# Comment out next line if not using pre-computed image features
image_path = "/saltpool0/data/layneberry/PlacesAudio_400k_distro/image_encodings_from_clip_large.pkl"

english_path = '/saltpool0/data/layneberry/PlacesAudio_400k_distro/PlacesEnglishSubset_test.json'
japanese_path = '/saltpool0/data/layneberry/PlacesAudioJpn_100k/PlacesJpn_test.json'
hindi_path = '/saltpool0/data/layneberry/hindi_places_100k/PlacesHindi_test.json'
val_all = DataLoader(Multilingual_Places_DataLoader_All(english_path, japanese_path, hindi_path, image_path),batch_size=100,shuffle=False,num_workers=16,drop_last=True)

model = th.nn.DataParallel(Parallel(heads=8, layers=1, batch=100, gpus=1, feat_trainable=True, weighted_sum=True, hubert_size='large', clip_size='large', use_langID=False)).cuda()
state_dict = th.load('/saltpool0/scratch/layneberry/CLIPC/trainableHuBERT_fromNoLangIDMultiBatches.pth')
try:
    model.load_state_dict(state_dict)
except:
    # Older state dicts need cleaning
    to_pop = []
    to_modify = []
    for k,v in state_dict.items():
        if '.clip' in k:
            to_pop.append(k)
        elif 'transform' in k and 'layers' not in k:
            to_modify.append(k)
    for k in to_pop:
        state_dict.pop(k)
    for k in to_modify:
        state_dict[k.replace('transform', 'transform.layers.0')] = state_dict.pop(k)
    model.load_state_dict(state_dict)

all_english_encodings = []
all_japanese_encodings = []
all_hindi_encodings = []
all_text_encodings = []
images = []

with th.no_grad():
    for batch in tqdm(val_all):
        image_out, english_out = model(batch['image'].cuda(), batch['eng_caption'].cuda())
        all_english_encodings.append(english_out.detach().cpu())
        _, hindi_out = model(batch['image'].cuda(), batch['hindi_caption'].cuda())
        all_hindi_encodings.append(hindi_out.detach().cpu())
        _, japanese_out = model(batch['image'].cuda(), batch['jpn_caption'].cuda())
        all_japanese_encodings.append(japanese_out.detach().cpu())

        images.append(image_out)
        tokens = clip.tokenize(batch['english_text']).cuda()
        text = clip_model.encode_text(tokens)
        all_text_encodings.append(text.detach().cpu().float())

english = th.cat(all_english_encodings, dim=0)
japanese = th.cat(all_japanese_encodings, dim=0)
hindi = th.cat(all_hindi_encodings, dim=0)
texts = th.cat(all_text_encodings, dim=0)
images = th.cat(images, dim=0).detach().cpu().float()

print('English Speech to Japanese Speech')
compute_metrics(th.matmul(english, japanese.t()))
print('English Speech to Hindi Speech')
compute_metrics(th.matmul(english, hindi.t()))
print('Japanese Speech to English Speech')
compute_metrics(th.matmul(japanese, english.t()))
print('Japanese Speech to Hindi Speech')
compute_metrics(th.matmul(japanese, hindi.t()))
print('Hindi Speech to English Speech')
compute_metrics(th.matmul(hindi, english.t()))
print('Hindi Speech to Japanese Speech')
compute_metrics(th.matmul(hindi, japanese.t()))

print('English Text to Image')
compute_metrics(th.matmul(texts, images.t()))
print('Image to English Text')
compute_metrics(th.matmul(images, texts.t()))

print('English Speech to English Text')
compute_metrics(th.matmul(english, texts.t()))
print('English Text to English Speech')
compute_metrics(th.matmul(texts, english.t()))
print('Japanese Speech to English Text')
compute_metrics(th.matmul(japanese, texts.t()))
print('English Text to Japanese Speech')
compute_metrics(th.matmul(texts, japanese.t()))
print('Hindi Speech to English Text')
compute_metrics(th.matmul(hindi, texts.t()))
print('English Text to Hindi Speech')
compute_metrics(th.matmul(texts, hindi.t()))
