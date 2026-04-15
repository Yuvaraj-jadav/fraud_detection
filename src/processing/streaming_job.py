from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from config.kafka_config import KAFKA_CONFIG, SPARK_CONFIG

spark = SparkSession.builder \
    .appName(SPARK_CONFIG["app_name"] + "_Streaming") \
    .master(SPARK_CONFIG["master"]) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


def get_schema():
    try:
        sample_df = spark.read \
            .format("kafka") \
            .option("kafka.bootstrap.servers", KAFKA_CONFIG["bootstrap_servers"]) \
            .option("subscribe", KAFKA_CONFIG["topic"]) \
            .option("startingOffsets", "earliest") \
            .option("endingOffsets", "latest") \
            .load()

        rows = sample_df.selectExpr("CAST(value AS STRING)").limit(1).collect()

        if len(rows) == 0:
            raise Exception("Kafka topic empty")

        import json
        sample_dict = json.loads(rows[0][0])

        from pyspark.sql.types import StructType, StructField

        schema = StructType([
            StructField(k, StringType(), True) for k in sample_dict.keys()
        ])

        print("[Streaming] Auto schema generated")
        return schema

    except Exception as e:
        print(f"[WARNING] Using fallback schema: {e}")

        # fallback schema (minimal)
        return StructType().add("value", StringType())


# Storage Paths
HDFS_OUTPUT_PATH = "/user/yuvaraj_hadoop/fraud_detection/data/processed/realtime/"
CHECKPOINT_PATH = "/user/yuvaraj_hadoop/fraud_detection/checkpoints/realtime/"

def process_batch(batch_df, batch_id):
    """
    Process each micro-batch: show on console and save to HDFS.
    """
    if not batch_df.isEmpty():
        print(f"\n[Processing Batch: {batch_id}]")
        # Display on Console
        batch_df.show(truncate=False)
        
        # Save to HDFS in Parquet format
        batch_df.write.format("parquet") \
            .mode("append") \
            .save(HDFS_OUTPUT_PATH)
        print(f"[Done] Batch {batch_id} persisted to: {HDFS_OUTPUT_PATH}")
    else:
        print(f"[Info] Batch {batch_id} is empty. Skipping storage.")

def run_streaming():
    schema = get_schema()

    raw_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_CONFIG["bootstrap_servers"]) \
        .option("subscribe", KAFKA_CONFIG["topic"]) \
        .option("startingOffsets", "earliest") \
        .load()

    parsed = raw_df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    # Use foreachBatch to send data to multiple sinks (Console + HDFS)
    query = parsed.writeStream \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", CHECKPOINT_PATH) \
        .start()

    query.awaitTermination()



if __name__ == "__main__":
    print("🚀 Starting Spark Streaming...")
    run_streaming()