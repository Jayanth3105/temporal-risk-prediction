# Learning-Based Temporal Motion Prediction for Risk-Aware Autonomous Driving

**RGB Camera Sequences → Temporal Deep Learning → Future Motion Prediction → Risk-Aware Visualization**

This project investigates learning-based future motion prediction for autonomous driving using sequential RGB camera observations. It compares two temporal architectures—**CNN → LSTM** and **CNN → BiLSTM**—and connects predicted trajectories to a simple risk-aware analysis.

---

## 🚀 Overview

- **Input:** Sequential RGB camera frames (Argoverse 2 Sensor Dataset)  
- **Pipeline:** Detection → Tracking → Trajectory History → ResNet-18 Features → LSTM/BiLSTM → Future Trajectories  
- **Output:** Predicted trajectories, ADE/FDE metrics, risk levels, and visualizations  

The core idea: *single-frame perception tells what is happening now; temporal prediction tells what is likely to happen next.*

---

## 🎯 Objectives

- Process sequential RGB observations and track objects over time  
- Build trajectory histories and extract visual features with ResNet-18  
- Model temporal dynamics using **LSTM** and **BiLSTM**  
- Predict multiple future trajectory points  
- Evaluate with **ADE** (Average Displacement Error) and **FDE** (Final Displacement Error)  
- Compare LSTM vs BiLSTM performance  
- Link future motion to a simple **risk assessment** (distance, speed, TTC)  
- Provide interpretable quantitative and visual results  

---

## 🧠 Models

| Model   | Temporal Processing                 |
|--------|-------------------------------------|
| LSTM   | Unidirectional recurrent modeling   |
| BiLSTM | Bidirectional recurrent modeling    |

Both models predict the same future trajectory points and are compared using the same metrics and visualizations.

---

## 📊 Evaluation

- **ADE:** Average error over all predicted time steps (lower = better)  
- **FDE:** Error at the final predicted time step (lower = better)  
- Additional analyses:  
  - Error vs prediction horizon  
  - Class-wise ADE  
  - ADE distribution  
  - Risk distribution  

---

## 🗂️ Repository Structure

```text
temporal-risk-prediction/
│
├── src/
│   ├── config.py
│   ├── dataset_builder.py
│   ├── detect.py
│   ├── track.py
│   ├── tracked_object.py
│   ├── trajectory.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── risk.py
│   ├── results_analyzer.py
│   ├── results_analyzer_bilstm.py
│   └── main.py
│
├── models/
│   ├── lstm/
│   └── bilstm/
│
├── outputs/
│   └── figures/
│
├── notebooks/
├── configs/
├── docs/
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Jayanth3105/temporal-risk-prediction.git
cd temporal-risk-prediction

python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

pip install -r requirements.txt
```

> The Argoverse 2 dataset must be downloaded and prepared separately.

---

## ▶️ Usage

Run the main pipeline:

```bash
python3 src/main.py
```

Configuration is controlled via `src/config.py`. Individual stages (dataset building, training, prediction, evaluation, risk analysis) are implemented as separate modules under `src/`.

---

## 📈 Outputs

The project produces:

- Trained models (LSTM / BiLSTM)  
- Trajectory predictions (JSON / internal formats)  
- Evaluation metrics (ADE, FDE, horizon-wise errors)  
- Risk analysis results  
- Plots and figures in `outputs/figures/`  
- Demo videos (distributed separately due to size):  
  - `result_lstm.mp4`  
  - `result_bilstm.mp4`  

---

## 🧪 Results

- Quantitative:  
  - LSTM vs BiLSTM ADE/FDE  
  - Error vs prediction horizon  
  - Class-wise ADE  
  - ADE and risk distributions  
- Qualitative:  
  - Observed vs predicted vs ground-truth trajectories  
  - Camera scenes with tracked objects and risk levels  

Numerical metrics are interpreted together with trajectory visualizations and risk analysis.

---

## ⚠️ Limitations

- Experimental research implementation, not a production ADAS system  
- Future motion is inherently uncertain; errors grow with horizon  
- Camera-based perception depends on scene and visibility conditions  
- Risk assessment is a simplified, research-oriented formulation  

---

## 🔮 Future Work

Possible extensions:

- Improved coordinate-frame handling  
- Longer prediction horizons  
- Additional temporal architectures (e.g., Transformers, attention)  
- Uncertainty-aware and multimodal trajectory prediction  
- More sophisticated risk modeling  
- Larger-scale evaluation and integration with planning/control  

---

## 📄 License & Citation

*(Add your license and citation info here if needed.)*

---

## 👤 Author

**Jayanth Narayana K**  
Kyushu Institute of Technology  

Project: *Learning-Based Temporal Motion Prediction for Risk-Aware Autonomous Driving*
