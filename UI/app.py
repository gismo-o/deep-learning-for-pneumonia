import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- 1. Model Mimarisi ve Yükleme Fonksiyonu ---
# Streamlit'in, ağırlıkları yükleyebilmesi için modelin "boş" yapısını bilmesi gerekir.
# Bu yüzden eğitim kodundaki model oluşturma fonksiyonunu buraya aynen kopyalıyoruz.
def build_resnet18(dropout=0.3):
    m = models.resnet18() # ÖNEMLİ: Önceden eğitilmiş ağırlıkları burada YÜKLEMİYORUZ.
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 2))
    return m

# @st.cache_resource, modeli her seferinde yeniden yüklemek yerine hafızada tutar, bu da uygulamayı hızlandırır.
@st.cache_resource
def load_model(model_path="UI/best_pneumonia_model.pth"):
    """Modeli oluşturur ve kayıtlı ağırlıkları yükler."""
    model = build_resnet18()
    # Ağırlıkları yüklüyoruz. 'map_location' ile CPU'da da çalışmasını garantileriz.
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Modeli tahmin moduna alıyoruz.
    return model

# Sınıf isimleri
class_names = ['NORMAL', 'PNEUMONIA']

# --- 2. Görüntü İşleme ve Tahmin Fonksiyonu ---
def predict(image_bytes, model):
    """Yüklenen görüntüyü işler ve tahmin yapar."""
    # Eğitimdeki 'eval_tfms' ile BİREBİR AYNI dönüşümleri uyguluyoruz.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    img = Image.open(image_bytes).convert('RGB')
    tensor = transform(img).unsqueeze(0) # Batch boyutu ekle: [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        prediction_idx = logits.argmax(1).item()
        prediction_label = class_names[prediction_idx]
        confidence = probabilities[prediction_idx].item()

    return prediction_label, confidence

# --- 3. Streamlit Arayüzü ---
st.set_page_config(layout="wide", page_title="Zatürre Teşhis Asistanı")
st.title("🩺 Göğüs Röntgeninden Zatürre Teşhis Asistanı")
st.write("---")

# Modeli yükle
model = load_model()

# Kullanıcıya dosya yükleme ekranı göster
uploaded_file = st.file_uploader("Lütfen bir göğüs röntgeni görüntüsü yükleyin...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Görüntüyü ve tahmini göstermek için ekranı ikiye böl
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Yüklenen Röntgen Görüntüsü", use_container_width=True)

    with col2:
        st.write("### 🤖 Model Analizi")
        with st.spinner('Teşhis yapılıyor, lütfen bekleyin...'):
            label, conf = predict(uploaded_file, model)

        st.success("Analiz Tamamlandı!")
        st.write(f"**Tahmin Edilen Durum:** `{label}`")
        st.write(f"**Güven Skoru:** `{conf:.2%}`")

        if label == 'PNEUMONIA':
            st.error("Model, zatürre belirtileri tespit etmiştir. Lütfen bir doktora danışınız.", icon="⚠️")
        else:
            st.success("Model, belirgin zatürre belirtileri tespit etmemiştir.", icon="✅")