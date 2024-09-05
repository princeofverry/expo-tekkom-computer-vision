from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("best.onnx")

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Mengatur jendela menjadi fullscreen
def set_fullscreen(window_name):
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Loop untuk menangkap frame dari webcam
while True:
    ret, frame = cap.read()  # Baca frame dari webcam
    
    if not ret:
        break
    
    # Deteksi objek pada frame
    results = model(frame)
    
    # Tampilkan hasil deteksi pada frame
    annotated_frame = results[0].plot()
    
    # Mengatur jendela menjadi fullscreen
    window_name = "YOLO Detection"
    set_fullscreen(window_name)
    
    # Menampilkan frame dengan deteksi
    cv2.imshow(window_name, annotated_frame)
    
    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan semua yang dibuka
cap.release()
cv2.destroyAllWindows()
