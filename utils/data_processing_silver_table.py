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

from pyspark.sql.functions import col, when, isnan, isnull, regexp_replace, trim, upper, lower
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_clickstream_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    """
    Process clickstream data for silver layer - clean and structure the data
    """
    # Load bronze data
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    
    if not os.path.exists(filepath):
        print(f"Bronze file not found: {filepath}")
        return None
    
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'{snapshot_date_str} silver clickstream processing - input rows:', df.count())
    
    # Data cleaning and validation
    # 1. Remove any rows with null Customer_ID or snapshot_date
    df = df.filter(col('Customer_ID').isNotNull() & col('snapshot_date').isNotNull())
    
    # 2. Clean feature columns - replace any null values with 0 for numerical features
    feature_cols = [f'fe_{i}' for i in range(1, 21)]
    for col_name in feature_cols:
        df = df.withColumn(col_name, when(col(col_name).isNull(), 0).otherwise(col(col_name)))
    
    # 3. Ensure Customer_ID is properly formatted
    df = df.withColumn('Customer_ID', trim(upper(col('Customer_ID'))))
    
    # 4. Ensure snapshot_date is properly formatted
    df = df.withColumn('snapshot_date', col('snapshot_date').cast(DateType()))
    
    print(f'{snapshot_date_str} silver clickstream processing - output rows:', df.count())
    
    # Save silver table as PARQUET
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df


def process_silver_attributes_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Process customer attributes data for silver layer - clean and structure the data
    Based on actual data analysis - handles Age as string, invalid SSN values
    """
    # Load bronze data - DO NOT infer schema due to Age being stored as string
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    
    if not os.path.exists(filepath):
        print(f"Bronze file not found: {filepath}")
        return None
    
    df = spark.read.csv(filepath, header=True, inferSchema=False)  # Read as strings first
    print(f'{snapshot_date_str} silver attributes processing - input rows:', df.count())
    
    # Data cleaning and validation
    # 1. Remove any rows with null Customer_ID or snapshot_date
    df = df.filter(col('Customer_ID').isNotNull() & col('snapshot_date').isNotNull())
    
    # 2. Clean Customer_ID
    df = df.withColumn('Customer_ID', trim(upper(col('Customer_ID'))))
    
    # 3. Clean Name - remove special characters and standardize
    df = df.withColumn('Name', trim(col('Name')))
    
    # 4. Clean Age - stored as string in source, convert to integer
    # Handle invalid ages by converting to null then to 0
    df = df.withColumn('Age', 
                      when(col('Age').isNull() | (col('Age') == ''), 0)
                      .otherwise(col('Age').cast(IntegerType())))
    df = df.withColumn('Age', 
                      when(col('Age').isNull(), 0)
                      .when((col('Age') < 0) | (col('Age') > 120), 0)
                      .otherwise(col('Age')))
    
    # 5. Clean SSN - mask sensitive data for privacy (SSN has garbage values like "#F%$D@*&8")
    df = df.withColumn('SSN', F.lit('MASKED'))
    
    # 6. Clean Occupation - standardize case and handle nulls
    df = df.withColumn('Occupation', 
                      when(col('Occupation').isNull(), 'Unknown')
                      .otherwise(trim(regexp_replace(col('Occupation'), '_', ' '))))
    
    # 7. Ensure snapshot_date is properly formatted
    df = df.withColumn('snapshot_date', col('snapshot_date').cast(DateType()))
    
    print(f'{snapshot_date_str} silver attributes processing - output rows:', df.count())
    
    # Save silver table as PARQUET
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df


def process_silver_financials_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    """
    Process customer financial data for silver layer - clean and structure the data
    Based on actual data analysis - handles string columns, underscores, and missing values
    """
    # Load bronze data - DO NOT infer schema to handle messy data
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    
    if not os.path.exists(filepath):
        print(f"Bronze file not found: {filepath}")
        return None
    
    df = spark.read.csv(filepath, header=True, inferSchema=False)  # Read as strings first
    print(f'{snapshot_date_str} silver financials processing - input rows:', df.count())
    
    # Data cleaning and validation
    # 1. Remove any rows with null Customer_ID or snapshot_date
    df = df.filter(col('Customer_ID').isNotNull() & col('snapshot_date').isNotNull())
    
    # 2. Clean Customer_ID
    df = df.withColumn('Customer_ID', trim(upper(col('Customer_ID'))))
    
    # 3. Clean Annual_Income - remove trailing underscores and convert to float
    df = df.withColumn('Annual_Income', 
                      regexp_replace(col('Annual_Income'), '_', ''))
    df = df.withColumn('Annual_Income', 
                      when(col('Annual_Income').isNull() | (col('Annual_Income') == ''), 0)
                      .otherwise(col('Annual_Income').cast(FloatType())))
    
    # 4. Clean Monthly_Inhand_Salary
    df = df.withColumn('Monthly_Inhand_Salary',
                      when(col('Monthly_Inhand_Salary').isNull(), 0)
                      .otherwise(col('Monthly_Inhand_Salary').cast(FloatType())))
    
    # 5. Clean integer columns
    int_cols = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Delay_from_due_date']
    for col_name in int_cols:
        df = df.withColumn(col_name,
                          when(col(col_name).isNull(), 0)
                          .otherwise(col(col_name).cast(IntegerType())))
    
    # 6. Clean Num_of_Loan - stored as string in source
    df = df.withColumn('Num_of_Loan',
                      when(col('Num_of_Loan').isNull() | (col('Num_of_Loan') == ''), 0)
                      .otherwise(col('Num_of_Loan').cast(IntegerType())))
    
    # 7. Clean Num_of_Delayed_Payment - stored as string
    df = df.withColumn('Num_of_Delayed_Payment',
                      when(col('Num_of_Delayed_Payment').isNull() | (col('Num_of_Delayed_Payment') == ''), 0)
                      .otherwise(col('Num_of_Delayed_Payment').cast(IntegerType())))
    
    # 8. Clean Changed_Credit_Limit - stored as string
    df = df.withColumn('Changed_Credit_Limit',
                      regexp_replace(col('Changed_Credit_Limit'), '_', ''))
    df = df.withColumn('Changed_Credit_Limit',
                      when(col('Changed_Credit_Limit').isNull() | (col('Changed_Credit_Limit') == ''), 0)
                      .otherwise(col('Changed_Credit_Limit').cast(FloatType())))
    
    # 9. Clean Num_Credit_Inquiries
    df = df.withColumn('Num_Credit_Inquiries',
                      when(col('Num_Credit_Inquiries').isNull(), 0)
                      .otherwise(col('Num_Credit_Inquiries').cast(FloatType())))
    
    # 10. Clean Outstanding_Debt - stored as string
    df = df.withColumn('Outstanding_Debt',
                      regexp_replace(col('Outstanding_Debt'), '_', ''))
    df = df.withColumn('Outstanding_Debt',
                      when(col('Outstanding_Debt').isNull() | (col('Outstanding_Debt') == ''), 0)
                      .otherwise(col('Outstanding_Debt').cast(FloatType())))
    
    # 11. Clean Credit_Utilization_Ratio
    df = df.withColumn('Credit_Utilization_Ratio',
                      when(col('Credit_Utilization_Ratio').isNull(), 0)
                      .otherwise(col('Credit_Utilization_Ratio').cast(FloatType())))
    
    # 12. Clean Total_EMI_per_month
    df = df.withColumn('Total_EMI_per_month',
                      when(col('Total_EMI_per_month').isNull(), 0)
                      .otherwise(col('Total_EMI_per_month').cast(FloatType())))
    
    # 13. Clean Amount_invested_monthly - stored as string
    df = df.withColumn('Amount_invested_monthly',
                      regexp_replace(col('Amount_invested_monthly'), '_', ''))
    df = df.withColumn('Amount_invested_monthly',
                      when(col('Amount_invested_monthly').isNull() | (col('Amount_invested_monthly') == ''), 0)
                      .otherwise(col('Amount_invested_monthly').cast(FloatType())))
    
    # 14. Clean Monthly_Balance - stored as string
    df = df.withColumn('Monthly_Balance',
                      regexp_replace(col('Monthly_Balance'), '_', ''))
    df = df.withColumn('Monthly_Balance',
                      when(col('Monthly_Balance').isNull() | (col('Monthly_Balance') == ''), 0)
                      .otherwise(col('Monthly_Balance').cast(FloatType())))
    
    # 15. Clean categorical columns
    # Credit_Mix - handle "_" as missing value
    df = df.withColumn('Credit_Mix', 
                      when((col('Credit_Mix').isNull()) | (col('Credit_Mix') == '_'), 'Unknown')
                      .when(col('Credit_Mix') == 'Bad', 'Bad')
                      .when(col('Credit_Mix') == 'Good', 'Good')
                      .when(col('Credit_Mix') == 'Standard', 'Standard')
                      .otherwise('Unknown'))
    
    # Payment_of_Min_Amount
    df = df.withColumn('Payment_of_Min_Amount',
                      when(col('Payment_of_Min_Amount').isNull(), 'Unknown')
                      .when(col('Payment_of_Min_Amount') == 'NM', 'No')
                      .otherwise(trim(upper(col('Payment_of_Min_Amount')))))
    
    # Payment_Behaviour
    df = df.withColumn('Payment_Behaviour',
                      when(col('Payment_Behaviour').isNull(), 'Unknown')
                      .otherwise(trim(col('Payment_Behaviour'))))
    
    # Type_of_Loan - handle missing values
    df = df.withColumn('Type_of_Loan',
                      when((col('Type_of_Loan').isNull()) | (col('Type_of_Loan') == ''), 'None')
                      .otherwise(trim(col('Type_of_Loan'))))
    
    # 16. Credit_History_Age - extract years as numerical value
    df = df.withColumn('Credit_History_Age_Years', 
                      when(col('Credit_History_Age').isNull(), 0)
                      .otherwise(
                          when(col('Credit_History_Age').contains('Years'), 
                               regexp_replace(col('Credit_History_Age'), ' Years.*', '').cast(IntegerType()))
                          .otherwise(0)))
    
    # 17. Ensure snapshot_date is properly formatted
    df = df.withColumn('snapshot_date', col('snapshot_date').cast(DateType()))
    
    print(f'{snapshot_date_str} silver financials processing - output rows:', df.count())
    
    # Save silver table as PARQUET
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df


def process_silver_lms_table(snapshot_date_str, bronze_lms_directory, silver_lms_directory, spark):
    """
    Process LMS loan daily data for silver layer (following Lab 2 approach)
    Calculates MOB (Months on Book) and DPD (Days Past Due) in silver layer
    """
    # Load bronze data
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    
    if not os.path.exists(filepath):
        print(f"Bronze file not found: {filepath}")
        return None
    
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'{snapshot_date_str} silver LMS processing - input rows:', df.count())
    
    # Data cleaning and validation
    # 1. Remove any rows with null Customer_ID, loan_id, or snapshot_date
    df = df.filter(col('Customer_ID').isNotNull() & 
                   col('loan_id').isNotNull() & 
                   col('snapshot_date').isNotNull())
    
    # 2. Clean Customer_ID
    df = df.withColumn('Customer_ID', trim(upper(col('Customer_ID'))))
    
    # 3. Clean loan_id
    df = df.withColumn('loan_id', trim(col('loan_id')))
    
    # 4. Clean numerical columns - handle nulls
    numerical_cols = ['tenure', 'installment_num', 'loan_amt', 'due_amt', 'paid_amt', 'overdue_amt', 'balance']
    
    for col_name in numerical_cols:
        df = df.withColumn(col_name, when(col(col_name).isNull(), 0).otherwise(col(col_name)))
        # Ensure non-negative values for amounts
        if col_name in ['loan_amt', 'due_amt', 'paid_amt', 'overdue_amt', 'balance']:
            df = df.withColumn(col_name, when(col(col_name) < 0, 0).otherwise(col(col_name)))
    
    # 5. Ensure dates are properly formatted
    df = df.withColumn('loan_start_date', col('loan_start_date').cast(DateType()))
    df = df.withColumn('snapshot_date', col('snapshot_date').cast(DateType()))
    
    # 6. Calculate MOB (Months on Book) - like Lab 2
    # MOB = months between loan_start_date and snapshot_date
    df = df.withColumn('mob', F.months_between(col('snapshot_date'), col('loan_start_date')).cast(IntegerType()))
    
    # 7. Calculate DPD (Days Past Due) - following professor's Lab 2 approach
    # Step 1: Calculate how many installments were missed
    df = df.withColumn("installments_missed", 
                      F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType()))
    df = df.fillna(0, subset=["installments_missed"])
    
    # Step 2: Calculate when the first payment was missed
    df = df.withColumn("first_missed_date", 
                      F.when(col("installments_missed") > 0, 
                            F.add_months(col("snapshot_date"), -1 * col("installments_missed")))
                      .cast(DateType()))
    
    # Step 3: Calculate actual days past due
    df = df.withColumn("dpd", 
                      F.when(col("overdue_amt") > 0.0, 
                            F.datediff(col("snapshot_date"), col("first_missed_date")))
                      .otherwise(0)
                      .cast(IntegerType()))
    
    print(f'{snapshot_date_str} silver LMS processing - output rows:', df.count())
    
    # Save silver table as PARQUET (like Lab 2)
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_lms_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df
