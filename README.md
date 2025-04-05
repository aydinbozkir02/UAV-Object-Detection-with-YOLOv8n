# YOLOv8 ile Nesne Takibi (Video Üzerinde)

Bu proje, **YOLOv8** nesne algılama modeli kullanılarak bir videoda nesne tespiti ve hedefe kilitlenme işlevlerini gerçekleştirmektedir. Özellikle uçak gibi belirli nesneleri tanıyıp, ekranda belirlenen iç bölgeye girdiklerinde hedef kilitleme mekanizması devreye girer.

## Özellikler

- YOLOv8 ile gerçek zamanlı nesne algılama
- Belirli sınıflar için hedef kilitleme (örnek: planes)
- İç bölgeye giren nesneler için zamanlayıcı
- Başarıyla takip edilen hedef için "Locked" bildirimi

## Gereksinimler

Bu projeyi çalıştırabilmek için aşağıdaki Python paketlerinin kurulu olması gerekir:

- `opencv-python`
- `ultralytics`
- `numpy`

Kurulum için:

```bash
pip install opencv-python numpy ultralytics

model = YOLO(r"C:\...path...\best.pt")
video_path = r"C:\...path...\video.mp4"

python YoloNesneTakip.py





---
