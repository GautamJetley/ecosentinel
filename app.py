import os, io, numpy as np, streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2, requests

# ---------- utils ----------
def load_image(file): return np.array(Image.open(file).convert("RGB"))
def to_pil(arr): return Image.fromarray(arr)

def resize_pair(a,b,max_w=768):
    h1,w1=a.shape[:2]; h2,w2=b.shape[:2]
    w=min(max_w,w1,w2)
    a=cv2.resize(a,(w,int(h1*w/w1))); b=cv2.resize(b,(w,int(h2*w/w2)))
    h=min(a.shape[0],b.shape[0])
    return a[:h,:w], b[:h,:w]

def synthetic_pair(w=640,h=400):
    before=np.zeros((h,w,3),dtype=np.uint8); before[:]=(60,110,60)
    for x in range(20,w,60):
        for y in range(20,h,60):
            cv2.rectangle(before,(x,y),(x+40,y+40),(35,140,35),-1)
    after=before.copy(); rng=np.random.default_rng(42)
    for x in range(20,w,60):
        for y in range(20,h,60):
            if rng.random()<0.35: cv2.rectangle(after,(x,y),(x+40,y+40),(120,120,120),-1)
    cv2.line(before,(0,h//2),(w,h//2+10),(50,80,155),8)
    cv2.line(after,(0,h//2+12),(w,h//2+20),(50,80,155),8)
    return before,after

def compute_change(before_rgb, after_rgb, thresh=0.22):
    g1=cv2.cvtColor(before_rgb,cv2.COLOR_RGB2GRAY)
    g2=cv2.cvtColor(after_rgb,cv2.COLOR_RGB2GRAY)
    score,diff=ssim(g1,g2,full=True); diff=1-diff
    diff_norm=cv2.normalize(diff,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    blur=cv2.GaussianBlur(diff_norm,(0,0),1.2)
    _,mask=cv2.threshold((blur/255.0).astype(np.float32),thresh,1.0,cv2.THRESH_BINARY)
    mask=(mask*255).astype(np.uint8); pct=100.0*(mask>0).sum()/mask.size
    heat=cv2.applyColorMap(diff_norm,cv2.COLORMAP_JET)
    overlay=cv2.addWeighted(after_rgb,0.7,heat,0.3,0); overlay[mask==0]_

