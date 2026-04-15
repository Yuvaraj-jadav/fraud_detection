import json
import logging
from typing import Dict, Any
import os
from pathlib import Path
from kafka import KafkaConsumer

# Import our custom Inference Engine
import sys
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)
from src.processing.inference_engine import InferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionConsumer:
    """Kafka consumer for fraud detection and real-time inference."""
    
    def __init__(
        self,
        bootstrap_servers: list = None,
        topic: str = 'transactions',  # Updated to match producer
        group_id: str = 'fraud-detection-group',
        auto_offset_reset: str = 'latest'  # Changed to latest for real-time monitoring
    ):
        self.bootstrap_servers = bootstrap_servers or ['localhost:9092']
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.consumer = None
        
        # Load the global model
        model_path = os.path.join(project_root, "models", "global_fraud_model.json")
        self.engine = InferenceEngine(model_path)
        
    def start(self):
        """Start the Kafka consumer."""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                enable_auto_commit=True
            )
            logger.info(f"Consumer started for topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to start consumer: {str(e)}")
            raise
    
    def consume_messages(self, callback=None):
        """Consume and predict fraud for messages in the topic."""
        if not self.consumer:
            self.start()
        
        try:
            logger.info("Real-time Fraud Monitor active. Waiting for transactions...")
            for message in self.consumer:
                try:
                    data: Dict[str, Any] = message.value
                    self._process_message(data)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer error: {str(e)}")
        finally:
            self.stop()
    
    def _process_message(self, data: Dict[str, Any]):
        """Analyze transaction for fraud using the global model."""
        source = data.get('source_file', 'unknown')
        
        # Perform Inference
        result, probability = self.engine.predict(data)
        
        # Log Result
        if result == "Fraud":
            logger.warning(f" [ALERT] FRAUD DETECTED - Source: {source} | Probability: {probability:.4f}")
            logger.warning(f" Data: {data}")
        elif result == "Genuine":
            logger.info(f" [OK] TRANSACTION GENUINE - Source: {source} | Probability: {probability:.4f}")
        else:
            logger.info(f" [?] UNKNOWN - Prediction failed for message: {data}")

    def stop(self):
        """Stop the Kafka consumer."""
        if self.consumer:
            self.consumer.close()
            logger.info("Consumer stopped")


if __name__ == "__main__":
    # Example usage
    consumer = FraudDetectionConsumer(
        bootstrap_servers=['localhost:9092'],
        topic='transactions',
        group_id='fraud-detection-realtime'
    )
    consumer.consume_messages()
