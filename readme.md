# Towards Early Fault Warning for EV Batteries: A Physics-Informed Deep Learning Approach to Voltage Prediction under Real Conditions
Here is the professional English version of the `README.md`.

This project is a deep learning-based framework designed for end-to-end time-series prediction of battery states. It utilizes historical sensor data (voltage, current, temperature) to predict statistical distribution parameters of the battery's future state.

The core architecture combines **Bi-LSTM** with **Multi-Head Attention** and incorporates **Physics-Informed Neural Networks (PINN)** principles. By utilizing a custom **Physics-Informed Loss**, the model ensures that predictions adhere to the physical laws of the battery Equivalent Circuit Model (ECM).

## 🚀 Key Features

* **Physics-Informed Loss**: Integrates data-driven MSE loss with physical constraints derived from the ECM equation.
  $$
  (V_{\text{terminal}} = V_{\text{oc}} - V_{p} - I R_{o})
  $$
  This improves interpretability and generalization.
*   **Advanced Architecture**: Features a Bidirectional LSTM (Bi-LSTM) for temporal feature extraction and Multi-Head Attention pooling to capture critical time-step dependencies.
*   **Robust Data Handling**:
    
    *   Integrated Kalman Filter for data smoothing (preprocessing).
    *   **Memory Efficient Dynamic Sequence Loading**: Efficiently handles large datasets.
    *   Automatic handling of non-continuous time segments (gap detection).
* **Multi-Task Learning**: Simultaneously predicts Terminal Voltage ($V{terminal}$), Open Circuit Voltage ($V{oc}$), Polarization Voltage ($V{p}$), and Ohmic Resistance ($R_o$).
*   **Derivative Regularization**: Includes first and second-order derivative losses to ensure smooth prediction curves and reduce jitter.

## 📂 Project Structure

The project follows a modular design for clarity and maintainability:

```text
project/
├── config.py           # [Config] Global hyperparameters, file paths, and physics weights
├── models.py           # [Model] Definition of the ImprovedBiLSTMAttn architecture
├── losses.py           # [Core] Custom PhysicsInformedLoss and Derivative Loss functions
├── data_utils.py       # [Data] Dataset class, data cleaning, segmentation, and DataLoader setup
├── utils.py            # [Utils] Metrics (RMSE, MRE) and helper functions
├── train.py            # [Main] Complete pipeline for training, validation, testing, and saving
├── functions.py        # [External] Low-level processing (e.g., Kalman Filter) - *User provided
└── README.md           # Project documentation
```

## 🛠️ Requirements

Recommended environment: Python 3.8+.

*   torch >= 1.10
*   pandas
*   numpy
*   scikit-learn
*   tqdm
*   joblib

Install dependencies via pip:
```bash
pip install torch pandas numpy scikit-learn tqdm joblib
```

## ⚙️ Configuration 

All adjustable parameters are centralized in `config.py`. You can modify experiments without touching the core logic.

**Key Parameters:**
*   **Paths**: `DATA_FILE`, `OUTPUT_DIR`
*   **Training**: `BATCH_SIZE`, `EPOCHS`, `LR` (Learning Rate)
*   **Time Windows**:
    *   `TIME_STEPS_IN`: Historical input steps 
    *   `TIME_STEPS_OUT`: Future prediction steps 
*   **Loss Weights **:
    *   `LAMBDA_MAIN`: Weight for the primary task (Terminal Voltage) 
    *   `LAMBDA_PHYSICS`: Weight for the physical equation constraint
    *   `LAMBDA_AUX`: Weight for auxiliary tasks ($V_{oc}, V_p, R_o$)

## 🏃‍♂️ Quick Start

1.  **Prepare Data**: Ensure your CSV data path is correctly set in `config.py`.
2.  **Run Training**:
    ```bash
    python train.py
    ```
3.  **Check Results**: Artifacts and logs will be saved in the parent directory of your data file.

## 📊 Outputs

Upon completion, the following files are generated:

*   `predictions_result.csv`: Contains ground truth vs. predicted values for the test set, aligned with timestamps.
*   `best_model_physics_v2.pth`: PyTorch model weights corresponding to the lowest validation loss.
*   `feature_scaler.pkl` / `target_scaler.pkl`: Scikit-learn scaler objects used for normalization.
*   **Console Output**: Real-time logging of RMSE and MRE (Mean Relative Error) metrics.

## 🧠 Physics-Informed Logic

In `losses.py`, the model minimizes not only the prediction error against labels but also the **Physics Consistency Loss**:

$$ Loss_{physics} = || V_{terminal}^{pred} - (V_{oc}^{pred} - V_{p}^{pred} - I^{pred} \cdot R_o^{pred}) ||^2 $$

**Variables:**
*   $V_{terminal}$: Terminal Voltage
*   $V_{oc}$: Open Circuit Voltage
*   $V_{p}$: Polarization Voltage
*   $I \cdot R_o$: Voltage drop due to Ohmic Resistance

By tuning `LAMBDA_PHYSICS` in `config.py`, you can control how strictly the model enforces these physical laws.

## 🤝 Contribution & License

Contributions via Issues or Pull Requests are welcome. This project is intended for academic research and educational purposes.
```
```