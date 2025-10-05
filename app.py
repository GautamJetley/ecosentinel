import os, io, numpy as np, streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2, requests

def load_image(file): return np.array(Image.open(file).convert("RGB"))
def to_pil(arr): return Image.fromarray(arr)

def resize_pair(a,b,max_w=768):
    h1,w1=a.shape[:2]; h2,w2=b.shape[:2]
    w=min(max_w,w1,w2); a=cv2.resize(a,(w,int(h1*w/w1))); b=cv2.resize(b,(w,int(h2*w/w2)))
    h=min(a.shape[0],b.shape[0]); return a[:h,:w], b[:h,:w]

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
    overlay=cv2.addWeighted(after_rgb,0.7,heat,0.3,0); overlay[mask==0]=after_rgb[mask==0]
    return score,pct,diff_norm,mask,overlay

def llama_report(ssim_score,pct,user_note=""):
    key=os.getenv("GROQ_API_KEY")
    if not key: return "Meta Llama summary skipped (set GROQ_API_KEY to enable)."
    prompt=f"""You are EcoSentinel. Summarize land-cover change between two satellite images.
- Similarity (SSIM): {ssim_score:.3f}
- Estimated % area changed: {pct:.1f}%
- Analyst note: {user_note or "N/A"}
Write 6–8 neutral, factual sentences and one 'Implication' line."""
    r=requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization":f"Bearer {key}"},
        json={"model":"llama-3.1-8b-instant",
              "messages":[{"role":"user","content":prompt}],
              "temperature":0.2,"max_tokens":320}, timeout=30)
    r.raise_for_status(); return r.json()["choices"][0]["message"]["content"].strip()

def cerebras_enhance(text):
    api=os.getenv("CEREBRAS_API_KEY")
    url=os.getenv("CEREBRAS_API_URL","https://api.cerebras.ai/v1/chat/completions")
    model=os.getenv("CEREBRAS_MODEL","llama3.1-8b")
    if not api: return None
    try:
        r=requests.post(url,headers={"Authorization":f"Bearer {api}"},
            json={"model":model,"temperature":0.2,"max_tokens":250,
                  "messages":[
                    {"role":"system","content":"You are a concise environmental risk analyst."},
                    {"role":"user","content":f"Refine this summary and append a short 'Mitigation & Next Steps' section:\n\n{text}"}]},
            timeout=30)
        r.raise_for_status(); return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(Cerebras enhancement skipped: {e})"

st.set_page_config(page_title="EcoSentinel — Environmental Change Detector", layout="wide")
st.title("🌍 EcoSentinel — AI Environmental Change Detector")
st.caption("Before/After satellite images → change map, % area impacted, and AI report. Meta (Llama) + Cerebras + Docker.")

cL,cR=st.columns(2)
with cL: f1=st.file_uploader("BEFORE image (jpg/png)", type=["jpg","jpeg","png"])
with cR: f2=st.file_uploader("AFTER image (jpg/png)", type=["jpg","jpeg","png"])
use_demo=st.toggle("Use built-in demo pair", value=not (f1 and f2))
note=st.text_input("Optional analyst note", "")

if use_demo: before,after=synthetic_pair()
else:
    if not (f1 and f2): st.info("Upload both images or enable demo."); st.stop()
    before,after=load_image(f1),load_image(f2)

before,after=resize_pair(before,after)
score,pct,diff_norm,mask,overlay=compute_change(before,after)

a,b,c=st.columns(3)
with a: st.subheader("Before"); st.image(to_pil(before), use_container_width=True)
with b: st.subheader("After"); st.image(to_pil(after), use_container_width=True)
with c: st.subheader("Change Overlay"); st.image(to_pil(overlay), use_container_width=True)

st.markdown(f"**SSIM:** {score:.3f}  |  **Estimated % area changed:** {pct:.1f}%")

buf=io.BytesIO(); Image.fromarray(cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)).save(buf,format="PNG")
st.download_button("⬇️ Download change mask", data=buf.getvalue(), file_name="change_mask.png", mime="image/png")

st.markdown("---")
if st.button("Generate AI Environmental Report"):
    with st.spinner("Generating Llama report..."):
        base=llama_report(score,pct,note)
    st.success(base)
    ench=cerebras_enhance(base)
    if ench:
        st.markdown("**Cerebras Enhanced Summary**")
        st.info(ench)
