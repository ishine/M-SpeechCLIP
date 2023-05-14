from torch.utils.data import Dataset
import json, random
import pickle as pkl
from PIL import Image
import soundfile as sf
import numpy as np
import clip
from transformers import Wav2Vec2Processor, HubertModel
import torch as th
import time, random

class Places_DataLoader(Dataset):
    """Monolingual Places dataset loader."""

    def __init__(self, data_path, image_path, clip_preprocess=None, language='English'):
        self.language = language
        self.data = json.load(open(data_path, 'rb'))['data']
        self.caption_length = 15 # measured in seconds

        # To switch to image encoding in the dataloader, uncomment the next line and comment the one after; this will slow down training but may be useful for some use cases, like a new test set
        # self.img_preprocess = clip_preprocess
        self.img_encodings = pkl.load(open(image_path,'rb'))
        
        self.caption_lookup = None
        if language != 'English' and ('val' in data_path or 'test' in data_path):
            # Retrieving the English caption is useful for speech-text retrieval experiments
            # If not testing speech-text and don't have English captions, just comment this out and remove the 'english_text' field from __getitem__'s return dict
            self.caption_lookup = json.load(open('/saltpool0/data/layneberry/PlacesAudio_400k_distro/english_caption_lookup_dict.json'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.language == 'English':
            langID = 0
            cap_path = '/saltpool0/data/layneberry/PlacesAudio_400k_distro/' + self.data[idx]['wav']
        elif self.language == 'Hindi':
            langID = 2
            cap_path = '/saltpool0/data/layneberry/hindi_places_100k/hindi_wavs/' + self.data[idx]['hindi_wav']
        elif self.language == 'Japanese':
            langID = 1
            cap_path = '/saltpool0/data/layneberry/PlacesAudioJpn_100k/' + self.data[idx]['wav']
        caption_audio, sr = sf.read(cap_path)
        assert(sr==16000) # sampling rate 16 kHz
        target_length = self.caption_length * 16000
        if len(caption_audio) > target_length:
            # Randomize start time
            diff = len(caption_audio) - target_length
            start = random.randint(0,diff)
            caption_audio = caption_audio[start:start+target_length] # Truncate
        elif len(caption_audio) < target_length:
            caption_audio = np.concatenate((caption_audio, [0.0 for _ in range(target_length - len(caption_audio))])) # Zero-pad
        
        # To switch to image encoding in the dataloader, uncomment the next 2 lines and comment the 4 after
        # image = Image.open('/saltpool0/data/layneberry/PlacesAudio_400k_distro/images/' + self.data[idx]['image'])
        # image = self.img_preprocess(image)
        if self.data[idx]['image'][0] == '/':
            image = self.img_encodings[self.data[idx]['image']]
        else:
            image = self.img_encodings['/'+self.data[idx]['image']]

        eng_text = ''
        if self.language == 'English':
            eng_text = self.data[idx]['asr_text']
        elif self.caption_lookup != None and self.data[idx]['image'][0] == '/':
            eng_text = self.caption_lookup[self.data[idx]['image']]
        elif self.caption_lookup != None:
            eng_text = self.caption_lookup['/'+self.data[idx]['image']]

        return {'caption':caption_audio, 'image':image, 'caption_path':cap_path, 'image_path':self.data[idx]['image'], 'langID':langID, 'english_text':eng_text}


class Multilingual_Places_DataLoader(Dataset):
    """
    Places multilingual dataset loader
    This dataloader assigns each spoken caption (in any language) an idx, so images each appear three times
    """

    def __init__(self, data_path_eng, data_path_jpn, data_path_hindi, image_path, clip_preprocess=None):
        self.data_eng = json.load(open(data_path_eng, 'rb'))['data']
        self.data_jpn = json.load(open(data_path_jpn, 'rb'))['data']
        self.data_hindi = json.load(open(data_path_hindi, 'rb'))['data']

        self.caption_length = 15 # measured in seconds
       
        # To switch to image encoding in the dataloader, uncomment the next line and comment the one after; this will slow down training but may be useful for some use cases, like a new test set
        # self.img_preprocess = clip_preprocess
        self.img_encodings = pkl.load(open(image_path,'rb'))

    def __len__(self):
        return len(self.data_eng) + len(self.data_jpn) + len(self.data_hindi)

    def __getitem__(self, idx):
        langID = 0 # First 100k indexes are all English captions
        if idx >= len(self.data_eng):
            langID = 1 # Next 100k are Japanese captions
        if idx >= len(self.data_eng) + len(self.data_jpn):
            langID = 2 # Final 100k are Hindi captions

        # To switch to image encoding in the dataloader, switch which definition of 'image' is commented out in each of the following three cases
        if langID == 0:
            caption_audio, sr = sf.read('/saltpool0/data/layneberry/PlacesAudio_400k_distro/' + self.data_eng[idx]['wav'])
            # image = Image.open('/saltpool0/data/layneberry/PlacesAudio_400k_distro/images/' + self.data_eng[idx]['image'])
            image = self.img_encodings['/'+self.data_eng[idx]['image']]
        elif langID == 1:
            caption_audio, sr = sf.read('/saltpool0/data/layneberry/PlacesAudioJpn_100k/' + self.data_jpn[idx-len(self.data_eng)]['wav'])
            # image = Image.open('/saltpool0/data/layneberry/PlacesAudio_400k_distro/images/' + self.data_jpn[idx-len(self.data_eng)]['image'])
            image = self.img_encodings[self.data_jpn[idx-len(self.data_eng)]['image']]
        elif langID == 2:
            caption_audio, sr = sf.read('/saltpool0/data/layneberry/hindi_places_100k/hindi_wavs/' + self.data_hindi[idx-len(self.data_eng)-len(self.data_jpn)]['hindi_wav'])
            # image = Image.open('/saltpool0/data/layneberry/PlacesAudio_400k_distro/images/' + self.data_hindi[idx-len(self.data_eng)-len(self.data_jpn)]['image'])
            image = self.img_encodings['/'+self.data_hindi[idx-len(self.data_eng)-len(self.data_jpn)]['image']]

        assert(sr==16000) # sampling rate 16 kHz
        
        target_length = self.caption_length * 16000
        if len(caption_audio) > target_length:
            # Randomize start time
            diff = len(caption_audio) - target_length
            start = random.randint(0,diff)
            caption_audio = caption_audio[start:start+target_length] # Truncate
        elif len(caption_audio) < target_length:
            caption_audio = np.concatenate((caption_audio, [0.0 for _ in range(target_length - len(caption_audio))])) # Zero-pad
        
        # Uncomment the following if not loading image features
        # image = self.img_preprocess(image)

        return {'caption':caption_audio, 'image':image, 'langID':langID}



class Multilingual_Places_DataLoader_All(Dataset):
    """
    Places multilingual dataset loader
    This dataloader assigns each image an idx, so has 3x fewer items than Multilingual_Places_DataLoader above
    Captions in all three languages are returned for the same image, and one is selected at random to be the 'random_caption' returned for that image, so that caption language randomization doesn't need to be done after batching
    """

    def __init__(self, data_path_eng, data_path_jpn, data_path_hindi, image_path, clip_preprocess=None):
        self.data_eng = json.load(open(data_path_eng, 'rb'))['data']
        self.data_jpn = json.load(open(data_path_jpn, 'rb'))['data']
        self.data_hindi = json.load(open(data_path_hindi, 'rb'))['data']

        # Sort by image path to line up the indexes
        self.data_eng = sorted(self.data_eng, key=lambda x: '/'+x['image'])
        self.data_jpn = sorted(self.data_jpn, key=lambda x: x['image'])
        self.data_hindi = sorted(self.data_hindi, key=lambda x: '/'+x['image'])

        self.caption_length = 15 # measured in seconds

        # To switch to image encoding in the dataloader, uncomment the next line and comment the one after; this will slow down training but may be useful for some use cases, like a new test set
        # self.img_preprocess = clip_preprocess
        self.img_encodings = pkl.load(open(image_path,'rb'))

    def __len__(self):
        assert(len(self.data_eng) == len(self.data_jpn))
        assert(len(self.data_eng) == len(self.data_hindi))
        return len(self.data_eng)

    def __getitem__(self, idx):
        # To switch to image encoding in the dataloader, uncomment the next 2 lines and comment the one after
        # image = Image.open('/saltpool0/data/layneberry/PlacesAudio_400k_distro/images/' + self.data_eng[idx]['image'])
        # image = self.img_preprocess(image)
        image = self.img_encodings['/'+self.data_eng[idx]['image']]

        eng_audio, _ = sf.read('/saltpool0/data/layneberry/PlacesAudio_400k_distro/'+self.data_eng[idx]['wav'])

        hindi_audio, _ = sf.read('/saltpool0/data/layneberry/hindi_places_100k/hindi_wavs/' + self.data_hindi[idx]['hindi_wav'])
        assert('/'+self.data_hindi[idx]['image'] == '/' + self.data_eng[idx]['image'])
        
        jpn_audio, _ = sf.read('/saltpool0/data/layneberry/PlacesAudioJpn_100k/' + self.data_jpn[idx]['wav'])
        assert(self.data_jpn[idx]['image'] == '/'+self.data_eng[idx]['image'])
        
        target_length = self.caption_length * 16000
        if len(eng_audio) > target_length:
            # Randomize start time
            diff = len(eng_audio) - target_length
            start = random.randint(0,diff)
            eng_audio = eng_audio[start:start+target_length] # Truncate
        elif len(eng_audio) < target_length:
            eng_audio = np.concatenate((eng_audio, [0.0 for _ in range(target_length - len(eng_audio))])) # Zero-pad
        if len(jpn_audio) > target_length:
            # Randomize start time
            diff = len(jpn_audio) - target_length
            start = random.randint(0,diff)
            jpn_audio = jpn_audio[start:start+target_length] # Truncate
        elif len(jpn_audio) < target_length:
            jpn_audio = np.concatenate((jpn_audio, [0.0 for _ in range(target_length - len(jpn_audio))])) # Zero-pad
        if len(hindi_audio) > target_length:
            # Randomize start time
            diff = len(hindi_audio) - target_length
            start = random.randint(0,diff)
            hindi_audio = hindi_audio[start:start+target_length] # Truncate
        elif len(hindi_audio) < target_length:
            hindi_audio = np.concatenate((hindi_audio, [0.0 for _ in range(target_length - len(hindi_audio))])) # Zero-pad

        r = random.randint(0,2)
        if r == 0:
            random_caption = eng_audio
        elif r == 1:
            random_caption = jpn_audio
        elif r == 2:
            random_caption = hindi_audio

        eng_text = self.data_eng[idx]['asr_text']

        return {'eng_caption':eng_audio, 'hindi_caption':hindi_audio, 'jpn_caption':jpn_audio, 'image':image, 'random_caption':random_caption, 'langID':r, 'english_text':eng_text}
