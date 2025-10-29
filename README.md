# Triton    
<img src="Media/Poze/Triton Logo.png" alt="Logo" width="120">        
[![Platform](https://img.shields.io/badge/platform-RaspberryPi4-blue?style=flat-square)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)]()
[![Drone](https://img.shields.io/badge/Drone-Robot+-blue?style=flat-square)]()
[![AI Model](https://img.shields.io/badge/YOLO11-XGBoost-orange?style=flat-square)]()
<img src="Media/Poze/QR code.jpg" alt="qr code" width="100">
> **Triton** — „ochiul deasupra mării” al salvamarilor. Un sistem AI + dronă care monitorizează mediul marin, detectează situații de risc de înec și poate declanșa intervenții rapide în mod autonom.

![](media/overview.jpg)  
*Video demonstrativ — va fi adăugat în `media/`*

---

## Cuprins
- [Descriere generală](#descriere-generală)
- [Platformă & biblioteci](#platformă--biblioteci)
- [Arhitectura software](#arhitectura-software)
- [Modele AI folosite](#modele-ai-folosite)
- [Flux operațional (overview)](#flux-operațional-overview)
- [Interfață web](#interfață-web)
- [Control dronă](#control-dronă)
- [Setup & rulare](#setup--rulare)
- [Structura proiectului](#structura-proiectului)
- [Demo — poze & filmări](#demo--poze--filmări)

---

## Descriere generală

**Triton** rulează **local pe Raspberry Pi 4B** și monitorizează continuu suprafața apei:
- **detectează persoane** și **situații de risc**;
- **evaluează comportamentul** (înot normal vs. posibil înec) pe ferestre temporale;
- **declanșează acțiuni automate**: alertă, marcarea poziției, comenzi pentru dronă (decolare, deplasare, orbitare), activarea unui mecanism de salvare (ex. eliberarea unui flotor).

Sistemul este **modular**, **low-latency** și **independent de cloud**, proiectat să funcționeze în **timp real** pe hardware cu resurse limitate.

---

## Platformă & biblioteci

- **Platformă:** Raspberry Pi 4B (captură video, inferență AI, interfață web, control dronă)
- **Ultralytics YOLO (detect/seg/pose):** detecție persoane/obiecte, segmentare zone de risc, estimare schelet (keypoints)
- **OpenCV:** captură cadre, prelucrări de imagine, encodare **JPEG** pentru streaming
- **Flask + Flask-SocketIO:** interfață web locală (HTTP/MJPEG) + status/comenzi în timp real (WebSocket)
- **XGBoost + scikit-learn:** clasificare comportamentală pe ferestre temporale (pipeline cu imputer/scaler)
- **DroneKit (MAVLink/Pixhawk):** mod **GUIDED** (takeoff, goto, orbit, land)
- **RPi.GPIO (PWM):** acționare servomotor (mecanism de salvare) și componente auxiliare
- **NumPy/Pandas:** trăsături temporale din keypoints, bufferizare/organizare dată

---

## Arhitectura software

Arhitectura este **multi-threaded**, cu **cozi thread-safe** între etape:
- **Captură** – citește continuu camera (USB / PiCamera2), atașează timestamp (și GPS dacă e disponibil)
- **AI** – fire dedicate pentru:
  - **YOLO Detect** (rulează continuu),
  - **YOLO Seg** (la nevoie/context),
  - **YOLO Pose** (doar când există persoană în cadru)
- **Feature-engine & clasificare** – construiește **ferestre** de ~30 cadre din keypoints **normalizați** și rulează **XGBoost**
- **Streaming** – servește **RAW** și **AI** prin **HTTP/MJPEG**
- **Control dronă** – procesează secvențial comenzile (coadă FIFO, mașină de stare internă)

Comunicarea dintre componente se face prin **`queue.Queue`** și **evenimente**; fiecare modul consumă/produce mesaje fără a bloca restul pipeline-ului.

---

## Modele AI folosite

- **YOLOv11 — Detect**  
  *Rol:* găsește rapid **persoane** (și opțional animale marine).  
  *Utilizare:* rulează permanent la rezoluție optimizată; **declanșează** Pose/Clasificare când e relevant.

- **YOLOv11 — Segmentare (Seg)**  
  *Rol:* evidențiază **zone cu risc** (ex. curenți de rupere, adâncime crescută) dacă există model antrenat.  
  *Utilizare:* produce **măști** pentru overlay și poate ajusta regulile de decizie („persoană + zonă riscantă”).

- **YOLOv11 — Pose**  
  *Rol:* extrage **scheletul** (17/34 keypoints) pentru persoana detectată.  
  *Utilizare:* normalizare la **pelvis/șold** (centrare) + **scalare** la distanța umeri (invarianță la scară), opțional **aliniere** pe umeri; trimite keypoints normalizați către feature-engine.

- **XGBoost — Clasificare comportamentală**  
  *Rol:* clasifică **ferestre** (~30 cadre ≈ 4–5 s) în **înot normal** vs. **posibil înec**.  
  *Trăsături (exemple):* viteze încheieturi (medie/var), % timp **mâini deasupra umerilor**, lungime braț + **asimetrie L/R**, variabilitatea **unghiului trunchiului**, **deplasarea pelvisului** (medie/var).  
  *Decizie:* probabilitate → **prag** + **histerezis** pentru stabilitate.

---

## Flux operațional (overview)

1. **Camera → cadre** intră în coada de captură  
2. **YOLO Detect** găsește persoana → **YOLO Pose** (și eventual **Seg**) pornesc context-dependent  
3. **Pose** → **keypoints normalizați** → **feature-engine** formează fereastra temporală  
4. **XGBoost** calculează scor de **posibil înec** → modulul de decizie trimite **alertă** și **comenzi** dronei  
5. **UI** afișează **RAW + AI**, status (GPS, mod zbor, baterie) și permite **override** operatorului

---

## Interfață web

- **Video live:** două fluxuri (RAW + AI) prin **HTTP/MJPEG**
- **Status live:** poziție, baterie, conexiune, mod zbor, stări AI (WebSocket)
- **Comenzi:** activare/dezactivare modele, confirmare alerte, comenzi dronă (decolare, goto, orbit, land)
- **Compatibilitate:** desktop / tabletă / telefon (acces local)

---

## Control dronă

- **Conectivitate:** Pixhawk via **MAVLink** (DroneKit)  
- **Moduri:** **GUIDED** (armare, `simple_takeoff`, `simple_goto`, orbit), **LAND** cu prioritate  
- **Acțiuni fizice:** servo (eliberare flotor), semnalizare vizuală/sonoră (opțional)

---

## Setup & rulare

**1) Dependențe (minim):**
- Python 3.11+, OpenCV, Ultralytics (YOLO), Flask, Flask-SocketIO (+eventlet/gevent), NumPy, Pandas, scikit-learn, XGBoost, DroneKit, RPi.GPIO, (Picamera2 pe Raspberry Pi)

**2) Pornire locală (exemplu):**
```bash
# 1. creează și activează venv
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 2. instalare dependențe
pip install -r requirements.txt

# 3. rulează serverul
python stream_app.py
# accesează UI în browser: http://<raspberrypi-ip>:<port>/
