import run as st
import cv2
import tempfile
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np
from ultralytics.utils.plotting import Annotator 

st.set_page_config(
    page_title="Helmet Detection",
    page_icon="🛵",
    layout="wide"
)

#judul
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Sistem Deteksi Pelanggaran Helm</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Tugas Besar Computer Vision - Deteksi Pengendara Tanpa Helm Berbasis YOLO11</p>", unsafe_allow_html=True)
st.divider()

#sidebar
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # 1. confidence level slider
    conf_thresh = st.slider(
        "Confidence Threshold", 
        min_value=0.0, max_value=1.0, value=0.35, step=0.05,
        help="Semakin tinggi, semakin sedikit kotak yang muncul (tapi lebih akurat)."
    )
    
    # 2. Slider IOU
    iou_thresh = st.slider(
        "IOU Threshold (Overlap)", 
        min_value=0.0, max_value=1.0, value=0.45, step=0.05,
        help="Mengatur tumpang tindih kotak deteksi."
    )
    
    st.write("---")
    source_type = st.radio("Pilih Sumber Data", ["Gambar (Upload)", "Video (Upload)"])

#load model yolo
@st.cache_resource
def load_model():
    # load model hasil training
    return YOLO('best.pt')

try:
    model = load_model()
    st.sidebar.success("✅ Model YOLOv11 Loaded!")
except:
    st.sidebar.error(" model tidak ditemukan ")

def predict_and_plot(media, is_video=False):
    # 1. Jalankan Prediksi
    if is_video:
        results = model.track(media, conf=conf_thresh, iou=iou_thresh, persist=True)
        img_array = media 
        color_no_helmet = (0, 0, 255) 
        color_helmet = (0, 255, 0)    
    else:
        results = model.predict(media, conf=conf_thresh, iou=iou_thresh)
        img_array = np.array(media)
        color_no_helmet = (255, 0, 0) 
        color_helmet = (0, 255, 0)     

    annotator = Annotator(img_array, line_width=3, example=str(model.names))

    # Hitung statistik
    no_helmet_count = 0
    helmet_count = 0
    
    # 3. Loop setiap kotak yang terdeteksi
    for box in results[0].boxes:
        # Ambil koordinat dan info kelas
        xyxy = box.xyxy[0]
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        
        label = f"{cls_name} {conf:.2f}"
        
        if "no_helmet" in cls_name:
            col = color_no_helmet
            no_helmet_count += 1
        elif "helmet" in cls_name and "no" not in cls_name:
            col = color_helmet
            helmet_count += 1
        else:
            col = (128, 128, 128)
        annotator.box_label(xyxy, label, color=col)
    return annotator.result(), no_helmet_count, helmet_count

# main page
if source_type == "Gambar (Upload)":
    uploaded_file = st.file_uploader("Upload Foto Jalan Raya", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Asli", use_container_width=True)
            
        if st.button("Deteksi Pelanggaran"):
            res_image, no_helm, helm = predict_and_plot(image)
            
            with col2:
                st.image(res_image, caption="Hasil Deteksi", use_container_width=True)
                
            # statistik
            st.write("Statistik")
            m1, m2, m3 = st.columns(3)
            m1.metric("Pengendara Tertib", f"{helm} Orang", delta_color="normal")
            m2.metric("PELANGGARAN", f"{no_helm} Orang", delta_color="inverse")
            
            if no_helm > 0:
                st.error(f"⚠️ Terdeteksi {no_helm} pelanggaran tidak memakai helm!")
            else:
                st.success("✅ Tidak ada pelanggaran terdeteksi.")

elif source_type == "Video (Upload)":
    uploaded_file = st.file_uploader("Upload Video CCTV", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        if st.button("Mulai Analisis Video"):
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            st_stat = st.empty()
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                
                # resize biar tidak berat saat renderrr
                frame = cv2.resize(frame, (800, 450))
              
                res_plotted, no_helm, helm = predict_and_plot(frame, is_video=True)
                
                # show playback
                frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update stats realtime
                with st_stat.container():
                    c1, c2 = st.columns(2)
                    c1.metric("Pelanggar (No Helmet)", no_helm)
                    c2.metric("Tertib (Helmet)", helm)
            
            vf.release()