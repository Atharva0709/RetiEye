# RetiEye
ResNet-50 classifier for 4 retinal OCT classes (CNV, DME, DRUSEN, NORMAL) with Grad-CAM and a Streamlit app.

**Educational prototype, not a medical device.**

## Method
- Data: Kaggle Retinal OCT (Kermany et al., 2018). Training images from patients that also appear in the official test set were excluded, then the rest was split 85/15 by patient.
- Preprocessing: grayscale, Gaussian blur, 3x3 median filter, resize 224x224, ImageNet normalization.
- Model: ImageNet-pretrained ResNet-50, 3 head epochs + 10 full fine-tuning epochs, AdamW, cosine schedule, class-weighted loss.
- Safeguards: "Uncertain, refer to specialist" below the confidence cutoff; rejection of non-OCT images.

## Results (official test set, 968 images)
Accuracy 98.66%, macro-F1 0.987, DME precision 0.996 / recall 0.983.

## Limitations
Single dataset, no external validation. One label per image. Some errors are made with high confidence.

## Run the app
pip install -r requirements.txt
streamlit run app/app.py

## Data citation
Kermany et al., Cell 2018. Please cite the original dataset paper.