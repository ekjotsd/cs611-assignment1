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
import argparse

from pyspark.sql.functions import col, when, isnan, isnull, sum as spark_sum, count, avg, max as spark_max, min as spark_min, first, last
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_gold_feature_store_table(snapshot_date_str, silver_clickstream_directory, silver_attributes_directory, 
                                   silver_financials_directory, silver_lms_directory, gold_feature_store_directory, spark):
    """
    Process all silver tables to create a unified feature store for ML model training
    This creates the gold layer feature store table
    """
    print(f"Processing gold feature store for {snapshot_date_str}")
    
    # Load all silver tables
    silver_tables = {}
    
    # 1. Load clickstream data
    clickstream_file = f"silver_clickstream_{snapshot_date_str.replace('-','_')}.parquet"
    clickstream_path = silver_clickstream_directory + clickstream_file
    if os.path.exists(clickstream_path):
        silver_tables['clickstream'] = spark.read.parquet(clickstream_path)
        print(f"Loaded clickstream data: {silver_tables['clickstream'].count()} rows")
    else:
        print(f"Clickstream file not found: {clickstream_path}")
        return None
    
    # 2. Load attributes data
    attributes_file = f"silver_attributes_{snapshot_date_str.replace('-','_')}.parquet"
    attributes_path = silver_attributes_directory + attributes_file
    if os.path.exists(attributes_path):
        silver_tables['attributes'] = spark.read.parquet(attributes_path)
        print(f"Loaded attributes data: {silver_tables['attributes'].count()} rows")
    else:
        print(f"Attributes file not found: {attributes_path}")
        return None
    
    # 3. Load financials data
    financials_file = f"silver_financials_{snapshot_date_str.replace('-','_')}.parquet"
    financials_path = silver_financials_directory + financials_file
    if os.path.exists(financials_path):
        silver_tables['financials'] = spark.read.parquet(financials_path)
        print(f"Loaded financials data: {silver_tables['financials'].count()} rows")
    else:
        print(f"Financials file not found: {financials_path}")
        return None
    
    # 4. Load LMS data - ONLY from MOB=0 (application time, installment_num=0)
    # This follows TA guidance: Feature store should contain features from loan application time
    # Labels come from MOB=6, and we inner join them later
    lms_file = f"silver_loan_daily_{snapshot_date_str.replace('-','_')}.parquet"
    lms_path = silver_lms_directory + lms_file
    if os.path.exists(lms_path):
        lms_df = spark.read.parquet(lms_path)
        print(f"Loaded LMS data: {lms_df.count()} rows")
        
        # CRITICAL: Only use installment_num = 0 (MOB=0, loan application time)
        # This is when the customer applies - we use ONLY info available at that moment
        lms_at_application = lms_df.filter(col('installment_num') == 0)
        print(f"Loans at application time (installment_num=0): {lms_at_application.count()} rows")
        
        # For each loan at application time, store loan characteristics
        # We keep loan_id to join with labels later (loan_id tracks the same loan over time)
        loan_features = lms_at_application.select(
            'loan_id',           # Key for joining with labels
            'Customer_ID',       # Customer identifier
            'loan_amt',          # Loan amount at origination
            'tenure',            # Planned loan tenure
            'loan_start_date',   # When loan started
            'snapshot_date'      # When this record was captured
        )
        
        silver_tables['loan_features'] = loan_features
        print(f"Created loan features from MOB=0: {silver_tables['loan_features'].count()} rows")
    else:
        print(f"LMS file not found: {lms_path}")
        # Create empty loan features if no LMS data
        silver_tables['loan_features'] = spark.createDataFrame([], "Customer_ID string, loan_id string")
    
    # Join all tables to create unified feature store
    # Strategy: Start with loans at MOB=0 (application time) and add customer features
    print("Joining all tables to create feature store...")
    
    # Start with loan features at MOB=0 (application time) as base
    # This ensures we only have loans that were just originated
    feature_store = silver_tables['loan_features']
    
    # Join with clickstream (behavioral features at application time)
    feature_store = feature_store.join(
        silver_tables['clickstream'],
        ['Customer_ID', 'snapshot_date'], 
        'left'
    )
    
    # Join with attributes (demographics at application time)
    feature_store = feature_store.join(
        silver_tables['attributes'].select('Customer_ID', 'Age', 'Occupation', 'snapshot_date'), 
        ['Customer_ID', 'snapshot_date'], 
        'left'
    )
    
    # Join with financials (financial profile at application time)
    feature_store = feature_store.join(
        silver_tables['financials'].select(
            'Customer_ID', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
            'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
            'Credit_Mix', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
            'Credit_History_Age_Years', 'Payment_of_Min_Amount', 'Total_EMI_per_month',
            'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance', 'snapshot_date'
        ), 
        ['Customer_ID', 'snapshot_date'], 
        'left'
    )
    
    # Fill null values in joined columns
    # For numerical columns, fill with 0
    numerical_columns = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age_Years',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
        'loan_amt', 'tenure'
        # NO TEMPORAL LEAKAGE: Features from installment_num=0 (MOB=0, application time)
        # Labels from installment_num=6 (MOB=6, 6 months later)
        # Inner join on loan_id ensures alignment
    ]
    
    for col_name in numerical_columns:
        if col_name in feature_store.columns:
            feature_store = feature_store.withColumn(col_name, when(col(col_name).isNull(), 0).otherwise(col(col_name)))
    
    # For categorical columns, fill with 'Unknown'
    categorical_columns = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
    
    for col_name in categorical_columns:
        if col_name in feature_store.columns:
            feature_store = feature_store.withColumn(col_name, when(col(col_name).isNull(), 'Unknown').otherwise(col(col_name)))
    
    # Add derived features for ML
    feature_store = feature_store.withColumn(
        'income_to_loan_ratio',
        when(col('Annual_Income') > 0, col('Total_EMI_per_month') * 12 / col('Annual_Income')).otherwise(0)
    )
    
    feature_store = feature_store.withColumn(
        'debt_to_income_ratio',
        when(col('Annual_Income') > 0, col('Outstanding_Debt') / col('Annual_Income')).otherwise(0)
    )
    
    feature_store = feature_store.withColumn(
        'credit_utilization_risk',
        when(col('Credit_Utilization_Ratio') > 80, 'High')
        .when(col('Credit_Utilization_Ratio') > 50, 'Medium')
        .otherwise('Low')
    )
    
    # Ensure snapshot_date is properly formatted
    feature_store = feature_store.withColumn('snapshot_date', col('snapshot_date').cast(DateType()))
    
    print(f'{snapshot_date_str} gold feature store - final rows:', feature_store.count())
    print(f'Feature store columns: {len(feature_store.columns)}')
    
    # Save gold feature store table as PARQUET (better for ML)
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    feature_store.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return feature_store


def process_gold_label_store_table(snapshot_date_str, silver_lms_directory, gold_label_store_directory, spark, dpd=30, mob=6):

    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table - Load parquet format
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_lms_directory + partition_name
    
    if not os.path.exists(filepath):
        print(f"Silver LMS file not found: {filepath}")
        return None
    
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())


    df = df.filter(col("mob") == mob)
    print(f'Loans at MOB={mob}: {df.count()} rows')

 
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    print(f'{snapshot_date_str} gold label store - output rows:', df.count())

    # save gold table as PARQUET (like Lab 2)
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df
