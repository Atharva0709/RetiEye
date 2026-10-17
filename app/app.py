import json, os
import cv2, numpy as np, torch, torch.nn as nn, torchvision, streamlit as st
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
INFO = {'CNV': 'Choroidal neovascularization (wet AMD)', 'DME': 'Diabetic macular edema',
        'DRUSEN': 'Drusen (early dry AMD)', 'NORMAL': 'Normal retina'}

def preprocess_img(gray):                       # identical to the training notebook
    g = cv2.GaussianBlur(gray, (0, 0), 1)
    g = cv2.medianBlur(g, 3)
    return cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA)

def to_tensor(img224):                          # identical to the training notebook
    x = np.repeat(img224[..., None], 3, axis=2).astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))

@st.cache_resource
def load():
    ck = torch.load(os.path.join(HERE, 'best_model.pt'), map_location=DEV)
    m = torchvision.models.resnet50(weights=None)
    m.fc = nn.Linear(2048, len(ck['classes']))
    m.load_state_dict(ck['state']); m = m.to(DEV).eval()
    store = {}
    m.avgpool.register_forward_hook(
        lambda mod, i, o: store.__setitem__('f', o.flatten(1)[0].detach().cpu().numpy().astype(np.float64)))
    thr = json.load(open(os.path.join(HERE, 'threshold.json')))['threshold']
    p, ood = os.path.join(HERE, 'ood_stats.npz'), None
    if os.path.exists(p):
        z = np.load(p); ood = (z['mu'].astype(np.float64), z['prec'].astype(np.float64), float(z['thr']))
    return m, ck['classes'], thr, ood, store

def ood_dist(f, ood):
    d = f[None, :] - ood[0]
    return float(np.sum((d @ ood[1]) * d, axis=1).min())

st.set_page_config(page_title='OCT Retinal Screening', layout='wide')
model, CLASSES, THR, OOD, STORE = load()
st.title('OCT retinal disease screening')
st.caption('Educational prototype, not a medical device. ResNet-50 trained on the Kaggle Retinal OCT dataset.')
st.sidebar.write(f'Uncertainty cutoff: {THR:.2f}')
st.sidebar.write('Classes: ' + ', '.join(f'{c} = {INFO[c]}' for c in CLASSES))
if OOD is None:
    st.sidebar.warning('ood_stats.npz not found: OCT-likeness check is OFF')

f = st.file_uploader('Upload an OCT scan (.jpg, .jpeg, .png)', type=['jpg', 'jpeg', 'png'])
if f:
    data = np.frombuffer(f.read(), np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    color = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if gray is None:
        st.error('Could not read this file as an image.'); st.stop()
    if min(gray.shape) < 128:
        st.error('Image is too small to be a valid OCT scan.'); st.stop()
    if float(np.abs(color[..., 0].astype(int) - color[..., 2].astype(int)).mean()) > 12:
        st.error('This is a colour image, not a grayscale OCT scan. No prediction made.'); st.stop()

    img = preprocess_img(gray)
    x = to_tensor(img).unsqueeze(0).to(DEV)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].cpu().numpy()
    if OOD is not None and ood_dist(STORE['f'], OOD) > OOD[2]:
        st.error('This does not look like a retinal OCT scan, so no prediction is made. Please upload an OCT B-scan.')
        st.image(img, caption='Uploaded image (after preprocessing)', clamp=True, width=300); st.stop()
    k = int(probs.argmax()); conf = float(probs[k])

    rgb = np.repeat((img / 255.0).astype(np.float32)[..., None], 3, axis=2)
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:   # needs gradients
        heat = cam(input_tensor=x, targets=None)[0]
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)

    c1, c2, c3 = st.columns(3)
    c1.image(img, caption='Preprocessed scan (224x224)', clamp=True, use_container_width=True)
    c2.image(overlay, caption='Grad-CAM: regions that most influenced the prediction', use_container_width=True)
    with c3:
        if conf < THR:
            st.warning('Uncertain, refer to specialist')
            st.write(f'Top guess (low confidence): {CLASSES[k]} ({conf*100:.1f}%)')
        else:
            st.success(f'Predicted: {CLASSES[k]} ({conf*100:.1f}% confidence)')
            st.write(INFO[CLASSES[k]])
        for c, p in zip(CLASSES, probs):
            st.progress(float(p), text=f'{c}: {p*100:.1f}%')