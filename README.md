# Big Data Analytics Fraud Detection System for Banking

A comprehensive, production-ready fraud detection system that implements advanced machine learning and big data analytics for real-time and batch processing of banking transactions. This project aligns with modern banking fraud prevention standards and the research by Imoisili Lucky Oseremen (2025) on transforming financial security through big data analytics.

---

## Project Overview

This system combines **big data engineering**, **machine learning ensembles**, and **real-time streaming** to detect fraudulent transactions in banking environments. It processes multi-source banking datasets, applies advanced feature engineering, and uses a 5-stage ML pipeline to identify fraud with high accuracy.

### Key Features

- **Multi-stage ML Ensemble Pipeline**
  - Stage 1: Isolation Forest + Logistic Regression (statistical anomaly detection)
  - Stage 2: XGBoost + SVM (pattern recognition)
  - Stage 3: Local Outlier Factor + One-Class SVM (unsupervised anomaly detection)
  - Stage 4: Autoencoder (reconstruction-error-based anomaly detection)
  - Stage 5: Behavioral LSTM (customer behavioral analysis)
  - Final: Ensemble decision logic (double-check with high-confidence thresholds)

- **Advanced Feature Engineering**
  - Temporal features: hour of day, day of week, weekend/night flags
  - Behavioral features: transaction frequency, customer/merchant relationships
  - Risk indicators: amount jumps, high-value transaction flags, IP/device mismatch
  - Location: city/country changes, fast geolocation changes
  - Relationship graph: merchant diversity, device diversity per customer

- **Big Data & Real-Time Processing**
  - Kafka-based transaction ingestion (3-broker cluster support)
  - PySpark distributed data cleaning and transformation
  - Spark Structured Streaming for real-time inference
  - Support for multiple banking datasets with automatic schema unification

- **Scalable, Production-Ready**
  - Virtual environment isolation
  - Modular architecture (each stage is independent)
  - Configuration-driven setup (Kafka, Spark settings)
  - Comprehensive metrics evaluation and reporting
  - Dashboard visualization option (Streamlit)

---

## Project Structure

```
fraud_detection/
├── README.md                          # This file
├── LAB_REPORT.md                      # Detailed paper alignment mapping
├── paper_alignment.md                 # Feature coverage checklist
├── project_explanation.txt            # Architectural deep-dive
├── commands.text                      # Quick-start commands
├── requirements.txt                   # Python dependencies
├── venv/                              # Python virtual environment
│
├── config/
│   └── kafka_config.py                # Kafka & Spark cluster config
│
├── data/
│   ├── raw/
│   │   ├── train/                     # Raw training datasets
│   │   ├── test/                      # Raw test datasets
│   │   └── streaming/                 # Live transaction staging
│   └── processed/                     # Unified, cleaned datasets
│
├── models/
│   ├── __init__.py                    # Pipeline exports
│   ├── pipeline.py                    # 5-stage ensemble orchestrator
│   ├── isolation_logistic.py          # Stage 1: Isolation Forest + LR
│   ├── xgb_svm.py                     # Stage 2: XGBoost + SVM
│   ├── oneclass_lof.py                # Stage 3: LOF + One-Class SVM
│   ├── autoencoder.py                 # Stage 4: TensorFlow Autoencoder
│   ├── behavior_lstm.py               # Stage 5: Behavioral analysis
│   └── global_fraud_model.json        # Pre-trained global model
│
├── src/
│   ├── ingestion/
│   │   ├── producer.py                # Kafka producer (CSV → Kafka)
│   │   └── consumer.py                # Kafka consumer (verification)
│   │
│   ├── processing/
│   │   ├── preprocessor.py            # Feature engineering + encoding
│   │   ├── unify_columns.py           # Dataset schema standardization
│   │   ├── train_global_model.py      # Global XGBoost training
│   │   ├── model_pipeline.py          # Batch inference orchestrator
│   │   ├── spark_job.py               # PySpark data cleaning
│   │   ├── streaming_job.py           # Spark Structured Streaming
│   │   ├── inference_engine.py        # Individual transaction prediction
│   │   ├── compare_algorithms.py      # Algorithm benchmarking
│   │   ├── compare_all_datasets.py    # Multi-dataset evaluation
│   │   └── metrics_report.py          # Performance reporting
│   │
│   └── query/                         # Reserved for Hive queries
│
├── results/
│   ├── plots/
│   │   ├── __init__.py
│   │   └── results.py                 # Metrics computation & visualization
│   └── reports/
│       └── algorithm_metrics.csv      # Generated metrics output
│
├── evaluate_metrics.py                # CLI evaluation runner
├── dashboard.py                       # Streamlit web dashboard
└── logs/                              # Application logs
```

---

## Quick Start

### 1. Setup Environment

```bash
cd ~/fraud_detection
source venv/bin/activate
./venv/bin/pip install -r requirements.txt
```

### 2. Prepare Data

**Download the Data & Pre-trained Models here:**
👉 [Google Drive: Fraud Detection Datasets & Models](https://drive.google.com/drive/folders/15NuQNq8OexMMbajzF1CgJoQsP1uzajOa?usp=sharing)

Extract the downloaded folders so that the raw datasets are located in `data/` and the pre-trained models are in `models/`. Then standardize the data:

```bash
./venv/bin/python src/processing/unify_columns.py
```

### 3. Train Global Model

```bash
./venv/bin/python src/processing/train_global_model.py
```

Output: `models/global_fraud_model.json`

### 4. Evaluate Pipeline

```bash
./venv/bin/python evaluate_metrics.py
```

Output: `results/reports/algorithm_metrics.csv`

### 5. Compare Across Datasets

```bash
./venv/bin/python src/processing/compare_all_datasets.py
```

### 6. Run Real-Time Streaming (Optional)

**Terminal 1 - Start Zookeeper:**
```bash
/opt/kafka/bin/zookeeper-server-start.sh /opt/kafka/config/zookeeper.properties
```

**Terminal 2 - Start Kafka Brokers:**
```bash
/opt/kafka/bin/kafka-server-start.sh /opt/kafka/config/server.properties
```

**Terminal 3 - Run Producer:**
```bash
./venv/bin/python src/ingestion/producer.py
```

**Terminal 4 - Run Streaming Job:**
```bash
spark-submit --master local[*] \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  src/processing/streaming_job.py
```

---

## Key Technologies

| Category | Technology | Purpose |
|----------|-----------|---------|
| ML/Data | pandas, scikit-learn, XGBoost | Feature engineering, model training |
| Deep Learning | TensorFlow, Keras | Autoencoder anomaly detection |
| Big Data | PySpark | Distributed processing |
| Streaming | Kafka | Real-time transaction ingestion |
| Visualization | Streamlit | Dashboard & reporting |
| Class Balancing | imbalanced-learn (SMOTE) | Handle fraud imbalance |

---

## Pipeline Architecture

```
Raw Transaction Data
        ↓
[Data Unification] → Standardized Schema
        ↓
[Preprocessor] → Feature Engineering + Encoding + Scaling
        ↓
[5-Stage ML Pipeline]
├─ Stage 1: Isolation Forest + Logistic Regression
│  └─ Output: anomaly_score, is_suspicious, fraud_probability
├─ Stage 2: XGBoost + SVM
│  └─ Output: xgb_score, svm_prediction, svm_probability
├─ Stage 3: LOF + One-Class SVM
│  └─ Output: lof_anomaly, ocsvm_anomaly, anomaly_ensemble
├─ Stage 4: Autoencoder
│  └─ Output: autoencoder_error, autoencoder_anomaly
├─ Stage 5: LSTM Behavioral Analysis
│  └─ Output: behavior_score, behavior_prediction
└─ Final Ensemble: Double-check logic
   └─ Output: ensemble_prediction, final_decision (Fraud/Genuine)
        ↓
[Metrics Evaluation]
        ↓
Results: Accuracy, Precision, Recall, F1-Score
```

---

## Performance Metrics

The system evaluates predictions across all stages:

- **Isolation Forest + Logistic**: Baseline anomaly detection
- **XGBoost + SVM**: Pattern-based supervised learning
- **Unsupervised Ensemble**: LOF & One-Class SVM anomaly combination
- **Autoencoder**: Reconstruction-based anomaly detection
- **Behavioral LSTM**: Customer behavior pattern deviation
- **Combined Ensemble**: Final high-confidence decision

See `results/reports/algorithm_metrics.csv` for detailed metrics.

---

## Configuration

### Kafka Settings (`config/kafka_config.py`)

```python
KAFKA_CONFIG = {
    "bootstrap_servers": "127.0.0.1:9092,127.0.0.1:9093,127.0.0.1:9094",
    "topic": "transactions",
}

SPARK_CONFIG = {
    "app_name": "fraud_detection",
    "master": "local[*]",
}
```

Adjust for your cluster setup.

---

## Feature Engineering Highlights

### Temporal Features
- Hour of day, day of week, week of year
- Is weekend, is night (00:00-05:59, 23:00-23:59)

### Behavioral Features
- Time delta since last transaction per customer
- Transaction frequency per customer
- Previous city, country, device, merchant

### Risk Indicators
- Amount delta and amount ratio (vs. previous transaction)
- Large amount jump (3x+ increase)
- High-value transaction (top 5%)
- Fast location change (< 1 hour, different city)

### Network/Graph Features
- IP device mismatch (IP used by multiple customers/devices)
- Customer unique merchants (diversity metric)
- Merchant unique customers
- Device diversity per customer/merchant

---

## Lab Report & Paper Alignment

For detailed mapping of this implementation to the research paper on big data analytics in banking fraud detection, see:

- **[LAB_REPORT.md](LAB_REPORT.md)** - Comprehensive feature-to-paper mapping
- **[paper_alignment.md](paper_alignment.md)** - Quick feature checklist

---

## Future Enhancements

- [ ] Explainability layer (SHAP, LIME)
- [ ] Regulatory compliance reporting (FFIEC guidelines)
- [ ] Graph-based fraud ring detection
- [ ] Federated learning for privacy-preserving multi-bank collaboration
- [ ] Containerization (Docker) for deployment
- [ ] API endpoint for real-time scoring

---

## Support & Questions

For questions about the implementation or architecture, refer to:
1. `project_explanation.txt` - High-level architecture
2. `paper_alignment.md` - Paper feature mapping
3. `LAB_REPORT.md` - Detailed report with claims evidence
4. Individual file docstrings in `models/` and `src/`

---

## License

This project is an academic implementation for research and educational purposes. All code is provided as-is for learning and lab submission.

**Reference Paper:** Imoisili Lucky Oseremen (2025). "Big Data Analytics in Fraud Detection and Prevention in Banking: Transforming Financial Security in the United States." *International Journal of Science, Engineering and Technology*, 13:6.

---

**Last Updated:** April 5, 2026
