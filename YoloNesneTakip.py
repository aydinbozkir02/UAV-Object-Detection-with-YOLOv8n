import cv2
import numpy as np
from ultralytics import YOLO
import time

# YOLOv8 modelini yükle
model = YOLO(r"C:\Users\bxnxa\Desktop\yolov8obb\runs\obb\train7\weights\best.pt")

# Video kaynağını aç
video_path = r"C:\Users/bxnxa\Desktop\yolov8obb\video2.mp4"  # Video dosya yolunu güncelleyin
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video açılamadı, dosya yolunu kontrol edin!", video_path)
    exit()

# Sayaç ve zamanlayıcı değişkenleri
detection_counter = 0
successful_detection_time = None
is_successful = False
success_display_time = None  # Başarı mesajı için zamanlayıcı
max_detection_time = 4  # 4 saniye boyunca sayaç çalışacak

# Sınıf isimleri
class_names = {0: "planes"}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Frame boyutlarını al
    height, width = frame.shape[:2]

    # Dikdörtgen alanını belirle (iç çerçeve)
    top = int(height * 0.1)
    bottom = int(height * 0.9)
    left = int(width * 0.25)
    right = int(width * 0.75)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Merkez noktayı hesapla
    center_x = width // 2
    center_y = height // 2

    # YOLOv8 ile nesne tespiti yap
    results = model(frame)

    detected_target = False  # Hedef tespiti durumu

    # Tespit edilen nesneleri işle
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # Normal bounding box (xyxy formatı)
        confidences = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confidences, classes):
            if conf > 0.5 and cls in class_names:  # Tanımlı sınıfları alıyoruz
                x1, y1, x2, y2 = map(int, box)
                
                # Bounding box çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Kutunun merkezini hesapla
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2
                
                # Merkez noktayı çiz
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Hedef merkezinden ana merkeze vektör çiz
                cv2.arrowedLine(frame, (box_center_x, box_center_y),
                                (center_x, center_y), (255, 0, 0), 2)
                
                # Güven skoru ve sınıf adını yaz
                label = f"{class_names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Eğer kutu iç çerçeve içinde ise sayaç başlat
                if left <= box_center_x <= right and top <= box_center_y <= bottom:
                    detected_target = True  # Hedef tespit edildi

    # Hedef tespit edilirse sayaç başlat
    if detected_target:
        if successful_detection_time is None:
            successful_detection_time = time.time()  # İlk tespit zamanı
    else:
        # Hedef tespit edilmezse sayaç sıfırlanır
        successful_detection_time = None

    # 4 saniye boyunca kesintisiz tespit yapılırsa başarılı olarak işaretle
    if successful_detection_time and time.time() - successful_detection_time >= max_detection_time:
        is_successful = True  # Başarı durumu

    # Eğer "Locked" yazısı ekranda görüntülenmesi gereken zaman geçmemişse, yazıyı göster
    if is_successful:
        if success_display_time is None:
            success_display_time = time.time()  # Başarı zamanını kaydet
        cv2.putText(frame, "Locked", (width // 2 - 50, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    # "Locked" yazısını 2 saniye sonra kaldır
    if success_display_time and time.time() - success_display_time >= 2:
        is_successful = False
        success_display_time = None  # Success display için zamanlayıcı sıfırlanır

    # Sayaç sadece 4 saniyeye kadar sayacak
    if successful_detection_time:
        elapsed_time = time.time() - successful_detection_time
        if elapsed_time <= max_detection_time:
            cv2.putText(frame, f"{elapsed_time:.2f}s", (left, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        else:
            cv2.putText(frame, f"{max_detection_time}s", (left, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Sonuçları göster
    cv2.imshow("YOLOv8 Tracking", frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizle
cap.release()
cv2.destroyAllWindows()
