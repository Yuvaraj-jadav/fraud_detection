#!/bin/bash

# Script to run the Fraud Detection project and capture outputs

echo "Starting Fraud Detection Project..."
echo "==================================="

# 1. Activate the virtual environment
echo "1. Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated."

# 2. Install Python dependencies (assuming already done, but run to ensure)
echo "2. Installing Python dependencies..."
./venv/bin/pip install -r requirements.txt
echo "Dependencies installed."

# 3. Start Zookeeper
echo "3. Starting Zookeeper..."
/home/yuvaraj_hadoop/kafka/bin/zookeeper-server-start.sh /home/yuvaraj_hadoop/kafka/config/zookeeper.properties &
sleep 5
echo "Zookeeper started."

# 4. Start Kafka brokers
echo "4. Starting Kafka brokers..."
/home/yuvaraj_hadoop/kafka/bin/kafka-server-start.sh /home/yuvaraj_hadoop/kafka/config/server.properties &
sleep 5
/home/yuvaraj_hadoop/kafka/bin/kafka-server-start.sh /home/yuvaraj_hadoop/kafka/config/server-1.properties &
sleep 5
/home/yuvaraj_hadoop/kafka/bin/kafka-server-start.sh /home/yuvaraj_hadoop/kafka/config/server-2.properties &
sleep 5
echo "Kafka brokers started."

# 5. Run the Kafka producer to create topic and stream data
echo "5. Running Kafka producer..."
./venv/bin/python src/ingestion/producer.py &
sleep 10  # Let it send some data
echo "Producer started."

# 6. Run the Kafka consumer in background
echo "6. Running Kafka consumer..."
./venv/bin/python src/ingestion/consumer.py &
sleep 5
echo "Consumer started."

# 7. Run Spark job for data processing
echo "7. Running Spark job..."
spark-submit --master local[*] src/processing/spark_job.py
echo "Spark job completed."

# 8. Run streaming job
echo "8. Running streaming job..."
timeout 30 spark-submit --master local[*] --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 src/processing/streaming_job.py
echo "Streaming job completed."

# 9. Train the global model
echo "9. Training global model..."
./venv/bin/python src/processing/train_global_model.py
echo "Model training completed."

# 10. Evaluate metrics
echo "10. Evaluating metrics..."
./venv/bin/python evaluate_metrics.py
echo "Metrics evaluation completed."

echo "==================================="
echo "Fraud Detection Project run complete!"