# OS 프로젝트(OBD) — Ultralytics YOLOv11 → TensorRT 고속 추론 툴

Windows 환경에서 **Ultralytics YOLO (YOLOv11)** 모델을 **TensorRT 엔진으로 변환**하고 **고속 추론·결과 저장**까지 수행하는 Python 기반 툴입니다.

Google Colab 없이, **Visual Studio Code + PowerShell**만으로 로컬 GPU/CPU에서 바로 실행할 수 있도록 구성되어 있습니다.

---

## 📦 프로젝트 개요

* **플랫폼:** Visual Studio Code · PowerShell (Python 3.11)
* **프레임워크:** Ultralytics YOLO (YOLOv11)
* **목적:** PyTorch `.pt` → TensorRT `.engine` 변환 및 FP16 GPU 추론
* **가속:** NVIDIA GPU (CUDA 12.1) + TensorRT 10.8
* **결과물:** TensorRT 엔진 파일(.engine), 예측 결과 이미지(자동 저장)

---

## ✅ 주요 기능

### 1. ⚙️ 원클릭 환경 구성

가상환경 생성(Python 3.11) → 필수 패키지 설치(`ultralytics`, `torch-cu121`, **TensorRT 10.8**, **ONNX/ORT**, **polygraphy**)까지 일괄 커맨드 제공

### 2. 🔍 GPU 인식 체크

`torch.cuda.is_available()` 및 장치명 출력으로 CUDA 사용 가능 여부 즉시 확인

### 3. 🚀 `.pt → .engine` 내보내기

Ultralytics `YOLO.export(format='engine', half=True, simplify=True, imgsz=640)`로 FP16 TensorRT 엔진 생성

### 4. 🖼️ 엔진 기반 단일 이미지 추론

생성된 `.engine`으로 고속 추론 수행
`save=True`로 결과 이미지 자동 저장

### 5. 🧩 유연한 옵션

`imgsz`, `half(True/False)`, `device('cuda'/CPU)`, `verbose` 등 핵심 파라미터 즉시 조정 가능

---

## 🧰 사용 방법

### 1) 프로젝트 루트 이동 & 가상환경 생성

```powershell
cd D:\TOYPROJECT\OS

# (3.11 고정) 가상환경 생성
py -3.11 -m venv .venv

# (권한 이슈 방지) 현재 세션만 실행 정책 완화
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 활성화
.\.venv\Scripts\Activate.ps1

# 확인
python -c "import sys; print(sys.executable)"
```

### 2) 필수 패키지 설치

```powershell
# pip 최신화
pip install --upgrade pip

# Ultralytics (YOLO) 설치/업데이트
pip install ultralytics --upgrade

# PyTorch CUDA 12.1 빌드 (GPU용)
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorRT + ONNX 관련 (엔진 변환/검증에 필요)
pip install --extra-index-url https://pypi.nvidia.com tensorrt==10.8.0
pip install onnx onnxruntime-gpu polygraphy
```

### 3) GPU 인식 확인

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"
```

### 4) `.pt → .engine` 내보내기

```powershell
python -c "from ultralytics import YOLO; \
m=YOLO(r'D:\TOYPROJECT\OS\Model\OBD.pt'); \
print(m.export(format='engine', device=0, half=True, simplify=True, imgsz=640))"
```

**주요 옵션**

* `imgsz=640` : 입력 해상도
* `half=True` : FP16(반정밀) 엔진으로 고속화
* `simplify=True` : 그래프 단순화로 호환성/성능 개선
* `device=0` : GPU 0번 사용 (CPU로 내보내려면 `device='cpu'`)

### 5) `.engine`으로 추론 테스트 (GPU + FP16)

```powershell
python -c "from ultralytics import YOLO; \
m=YOLO(r'D:\TOYPROJECT\OS\Model\OBD.engine'); \
m.predict(r'C:\Users\제이스텍\Desktop\영상테스트\ORG\20250721133834\20250721133834_CAM1_TAB2_OK.bmp', \
device='cuda', half=True, imgsz=640, save=True, verbose=True)"
```

* 실행 후 **결과 이미지**가 자동으로 저장됩니다(기본 `runs/predict` 하위).

---

## 🧩 경로 구조 예시

```
D:\TOYPROJECT\OS\
 ├─ Model\
 │   ├─ OBD.pt          # 학습된 PyTorch 가중치
 │   └─ OBD.engine      # 내보낸 TensorRT 엔진
 └─ ...                 # 소스/유틸 등
```

---

## 🔧 개발 환경 및 라이브러리

| 구성 요소      | 내용                                       |
| ---------- | ---------------------------------------- |
| 언어         | Python 3.11                              |
| 프레임워크      | Ultralytics YOLO (YOLOv11)               |
| 딥러닝 런타임    | PyTorch (CUDA 12.1 빌드)                   |
| 가속/엔진      | NVIDIA TensorRT 10.8 (FP16 지원)           |
| 모델 변환      | Ultralytics Export API, ONNX, Polygraphy |
| 추론 런타임(옵션) | onnxruntime-gpu                          |
| 실행 환경      | Visual Studio Code, Windows 10/11        |
| 쉘          | PowerShell                               |

---

## 💡 팁 & 주의사항

* **버전 호환**: `torch(cu121)` · **CUDA 12.1** · **TensorRT 10.8** 구성이 서로 맞아야 합니다.
* **FP16 정확도**: `half=True`는 속도 향상에 유리하나, 일부 모델/하드웨어에서 미세한 정확도 차이가 날 수 있습니다.
* **입력 해상도**: `imgsz`는 학습/내보내기/추론에서 일관되게 사용할수록 성능이 안정적입니다.
* **경로에 공백/한글**이 포함된 경우도 `r'raw-string'` 경로 표기면 안전하게 처리됩니다.
