import torch as th
import numpy as np
import clip
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2Model, HubertModel
import soundfile as sf
import torch.nn as nn
import pickle as pkl
import time
from tdl import TDL, TEL, Layered_TEL

class Parallel(th.nn.Module):
    def __init__(self, heads=1, layers=2, batch=4, gpus=4, feat_trainable=False, weighted_sum=False, hubert_size='large', clip_size='large', use_langID=False):
        super(Parallel, self).__init__()
        
        self.heads = heads
        self.layers = layers
        self.batch = batch // gpus
        self.gpus = gpus
        self.feat_train = feat_trainable
        self.weighted_sum = weighted_sum
        self.use_langID = use_langID
        self.hubert_size = hubert_size # Needed for getting the right attention layer

        if hubert_size == 'base':
            self.feat_dim = 768
            self.wav2vec = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states = True)
            if use_langID:
                self.feat_layer_weights = nn.Parameter(th.ones(3,13)/13.)
            else:
                self.feat_layer_weights = nn.Parameter(th.ones(13)/13.)
        else:
            self.feat_dim = 1024
            self.wav2vec = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
            self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft", output_hidden_states = True)
            if use_langID:
                self.feat_layer_weights = nn.Parameter(th.ones(3,25)/25.)
            else:
                self.feat_layer_weights = nn.Parameter(th.ones(25)/25.)
        
        if clip_size == 'base':
            self.out_size = 512
        else:
            self.out_size = 768

        self.transform = Layered_TEL(self.feat_dim, nhead=self.heads, nlayers=self.layers)
        self.token_to_clip = nn.Linear(self.feat_dim,self.out_size) # project predicted token into CLIP encoding

        """
        # CLIP model only needed if image features aren't pre-dumped
        if clip_size == 'base':
            self.clip, _ = clip.load("ViT-B/32")
        else:
            self.clip, _ = clip.load("ViT-L/14")
        self.clip.cuda()
        for p in self.clip.parameters():
            p.requires_grad = False
        """

        self.target_CLS = nn.Parameter(th.rand(self.feat_dim))
        if self.use_langID:
            self.lang_tokens = nn.Parameter(th.rand((3,self.feat_dim))) # 3 languages, one token each, access by indexing

    def weight_by(self, x, y, langID=None):
        x = th.nn.functional.softmax(x,dim=-1)
        if self.use_langID:
            x1 = x[0].repeat(list(y.shape[1:])+[1]).permute(3,0,1,2)
            x2 = th.stack([x for _ in range(self.batch)])
            by_batch = x2[th.arange(self.batch),langID].t().unsqueeze(2).unsqueeze(3)
            rep = by_batch.expand(-1,-1,y.shape[2],y.shape[3])
            return rep*y
        else:
            return x.repeat(list(y.shape[1:])+[1]).permute(3,0,1,2)*y

    def forward(self, img, audio, return_attentions=False, langID=None, return_feats_with_attns=False):
        if self.feat_train:
            with th.no_grad():
                feat_in = self.wav2vec(audio, sampling_rate=16000, return_tensors='pt')['input_values'].squeeze(0).cuda().type(self.hubert.dtype)
            feat_hs = self.hubert(feat_in).hidden_states
            feat_stacked = th.stack(feat_hs)
        else:
            with th.no_grad():
                feat_in = self.wav2vec(audio, sampling_rate=16000, return_tensors='pt')['input_values'].squeeze(0).cuda().type(self.hubert.dtype)
                feat_hs = self.hubert(feat_in).hidden_states
                feat_stacked = th.stack(feat_hs) 
        if self.weighted_sum:
            # we want a gradient for the weights!
            feat_weighted = self.weight_by(self.feat_layer_weights, feat_stacked, langID)
            feat_out = th.sum(feat_weighted, dim=0)
            feat_out = feat_out.permute(1,0,2)
        else:
            feat_out = feat_stacked[-1].permute(1,0,2)
       
        target = th.stack([self.target_CLS for _ in range(self.batch)]).unsqueeze(0)
        if self.use_langID:
            assert(langID != None)
            toks = th.stack([self.lang_tokens for _ in range(self.batch)])
            lang_toks = toks[th.arange(self.batch),langID].unsqueeze(0)
            feat_out = th.cat((lang_toks, feat_out),dim=0)
        if return_attentions or return_feats_with_attns:
            enc_in = th.cat((target, feat_out),dim=0)
            new_token, attn_weights = self.transform(enc_in, return_attentions=True)
        else:
            enc_in = th.cat((target, feat_out),dim=0)
            new_token = self.transform(enc_in)
        all_text_feats = new_token
        new_token = new_token[0]
        new_embd = self.token_to_clip(new_token).squeeze()
        if img != None:
            if len(img.shape) > 2:
                image_features = self.clip.encode_image(img)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            else:
                image_features = img # Using pre-computed features!
            if return_attentions and not return_feats_with_attns:
                return image_features, new_embd, attn_weights
            elif return_feats_with_attns:
                return image_features, new_embd, attn_weights, all_text_feats
            else:
                return image_features, new_embd
        else:
            if return_feats_with_attns:
                return new_embd, attn_weights, all_text_feats
            elif return_attentions:
                return new_embd, attn_weights
            else:
                return new_embd
