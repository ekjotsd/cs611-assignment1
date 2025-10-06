import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("CS611_Assignment1_FeatureStore") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# Generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print("Processing dates:", dates_str_lst)

# Create datamart directory structure
print("Creating datamart directory structure...")

# Bronze layer directories
bronze_clickstream_directory = "datamart/bronze/clickstream/"
bronze_attributes_directory = "datamart/bronze/attributes/"
bronze_financials_directory = "datamart/bronze/financials/"
bronze_lms_directory = "datamart/bronze/lms/"

# Silver layer directories
silver_clickstream_directory = "datamart/silver/clickstream/"
silver_attributes_directory = "datamart/silver/attributes/"
silver_financials_directory = "datamart/silver/financials/"
silver_lms_directory = "datamart/silver/lms/"

# Gold layer directories
gold_feature_store_directory = "datamart/gold/feature_store/"
gold_label_store_directory = "datamart/gold/label_store/"

# Create all directories
directories = [
    bronze_clickstream_directory, bronze_attributes_directory, bronze_financials_directory, bronze_lms_directory,
    silver_clickstream_directory, silver_attributes_directory, silver_financials_directory, silver_lms_directory,
    gold_feature_store_directory, gold_label_store_directory
]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

print("=== STARTING BRONZE LAYER PROCESSING ===")
# Run bronze backfill for all data sources
for date_str in dates_str_lst:
    print(f"\nProcessing bronze layer for {date_str}")
    
    # Process clickstream data
    utils.data_processing_bronze_table.process_bronze_clickstream_table(
        date_str, bronze_clickstream_directory, spark
    )
    
    # Process attributes data
    utils.data_processing_bronze_table.process_bronze_attributes_table(
        date_str, bronze_attributes_directory, spark
    )
    
    # Process financials data
    utils.data_processing_bronze_table.process_bronze_financials_table(
        date_str, bronze_financials_directory, spark
    )
    
    # Process LMS data
    utils.data_processing_bronze_table.process_bronze_lms_table(
        date_str, bronze_lms_directory, spark
    )

print("=== STARTING SILVER LAYER PROCESSING ===")
# Run silver backfill for all data sources
for date_str in dates_str_lst:
    print(f"\nProcessing silver layer for {date_str}")
    
    # Process clickstream data
    utils.data_processing_silver_table.process_silver_clickstream_table(
        date_str, bronze_clickstream_directory, silver_clickstream_directory, spark
    )
    
    # Process attributes data
    utils.data_processing_silver_table.process_silver_attributes_table(
        date_str, bronze_attributes_directory, silver_attributes_directory, spark
    )
    
    # Process financials data
    utils.data_processing_silver_table.process_silver_financials_table(
        date_str, bronze_financials_directory, silver_financials_directory, spark
    )
    
    # Process LMS data
    utils.data_processing_silver_table.process_silver_lms_table(
        date_str, bronze_lms_directory, silver_lms_directory, spark
    )

print("=== STARTING GOLD LAYER PROCESSING ===")
# Run gold backfill for feature store and label store
for date_str in dates_str_lst:
    print(f"\nProcessing gold layer for {date_str}")
    
    # Process feature store (combines all silver tables)
    utils.data_processing_gold_table.process_gold_feature_store_table(
        date_str, silver_clickstream_directory, silver_attributes_directory,
        silver_financials_directory, silver_lms_directory, gold_feature_store_directory, spark
    )
    
    # Process label store (for loan default prediction)
    utils.data_processing_gold_table.process_gold_label_store_table(
        date_str, silver_lms_directory, gold_label_store_directory, spark, dpd=30, mob=6
    )

print("=== PIPELINE COMPLETED ===")
print("Checking final results...")

# Check feature store results
feature_store_files = [gold_feature_store_directory + f for f in os.listdir(gold_feature_store_directory) if os.path.isdir(gold_feature_store_directory + f)]
if feature_store_files:
    print(f"Feature store files created: {len(feature_store_files)}")
    # Load and display summary of feature store
    df_feature_store = spark.read.parquet(*feature_store_files)
    print(f"Feature store total rows: {df_feature_store.count()}")
    print(f"Feature store columns: {len(df_feature_store.columns)}")
    print("Feature store schema:")
    df_feature_store.printSchema()


# Check label store results
label_store_files = [gold_label_store_directory + f for f in os.listdir(gold_label_store_directory) if os.path.isdir(gold_label_store_directory + f)]
if label_store_files:
    print(f"\nLabel store files created: {len(label_store_files)}")
    # Load and display summary of label store
    df_label_store = spark.read.parquet(*label_store_files)
    print(f"Label store total rows: {df_label_store.count()}")
    print("Label store schema:")
    df_label_store.printSchema()

    
    # Show label distribution
    print("\nLabel distribution:")
    df_label_store.groupBy('label').count().show()

print("=== DATAMART STRUCTURE CREATED ===")
print("Bronze layer: Raw data ingestion")
print("Silver layer: Cleaned and structured data")
print("Gold layer: Feature store and label store for ML")
print("Pipeline execution completed successfully!")

# Stop Spark session
spark.stop()
