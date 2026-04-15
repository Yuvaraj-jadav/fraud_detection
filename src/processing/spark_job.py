from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("FraudDetection") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Read sample data from CSV
df = spark.read.csv("file:///home/yuvaraj_hadoop/fraud_detection/data/raw/test/Transactions_data.csv", header=True, inferSchema=True)

# Feature engineering
df = df.withColumn("high_amount", df["amount"] > 5000)

df.show()

# Save processed data to Kafka topic
df.selectExpr("to_json(struct(*)) AS value") \
    .write \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "fraud_processed") \
    .save()
