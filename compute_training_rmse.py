#!/usr/bin/env python3
"""
Compute RMSE on training dataset using leave-one-out cross-validation.
"""

import json
import os
import csv
import numpy as np
from a4_custom_graph import (
    load_erd_data, load_grades, erd_similarity, 
    knn_predict, K, q_threshold
)

def compute_training_rmse():
    """Compute RMSE on training data using leave-one-out cross-validation."""
    
    print("Loading training data...")
    train_dataset1_dir = 'for_students/Dataset1'
    train_dataset2_dir = 'for_students/Dataset2'
    grades_csv = 'for_students/ERD_grades.csv'
    
    train_erds1 = load_erd_data(train_dataset1_dir)
    train_erds2 = load_erd_data(train_dataset2_dir)
    grades = load_grades(grades_csv)
    
    print(f"Loaded {len(train_erds1)} training ERDs from Dataset1")
    print(f"Loaded {len(train_erds2)} training ERDs from Dataset2")
    
    # Dataset 1 predictions
    print("\nComputing predictions for Dataset1 (leave-one-out)...")
    predictions1 = []
    actuals1 = []
    
    train_data1 = []
    train_grades1 = []
    train_erd_nums1 = []
    
    for erd_num, erd_data in train_erds1.items():
        if erd_num in grades and grades[erd_num][0] is not None:
            train_data1.append(erd_data)
            train_grades1.append(grades[erd_num][0])
            train_erd_nums1.append(erd_num)
    
    print(f"Processing {len(train_data1)} ERDs for Dataset1...", flush=True)
    for i in range(len(train_data1)):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(train_data1)}...", flush=True)
        
        # Leave-one-out: use all other ERDs for training
        test_erd = train_data1[i]
        actual_grade = train_grades1[i]
        
        # Create training set without current ERD
        train_erds_loo = train_data1[:i] + train_data1[i+1:]
        train_grades_loo = train_grades1[:i] + train_grades1[i+1:]
        
        # Predict
        predicted_grade = knn_predict(test_erd, train_erds_loo, train_grades_loo, k=K, threshold=q_threshold)
        
        predictions1.append(predicted_grade)
        actuals1.append(actual_grade)
    
    # Dataset 2 predictions
    print("\nComputing predictions for Dataset2 (leave-one-out)...")
    predictions2 = []
    actuals2 = []
    
    train_data2 = []
    train_grades2 = []
    train_erd_nums2 = []
    
    for erd_num, erd_data in train_erds2.items():
        if erd_num in grades and grades[erd_num][1] is not None:
            train_data2.append(erd_data)
            train_grades2.append(grades[erd_num][1])
            train_erd_nums2.append(erd_num)
    
    print(f"Processing {len(train_data2)} ERDs for Dataset2...", flush=True)
    for i in range(len(train_data2)):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(train_data2)}...", flush=True)
        
        # Leave-one-out: use all other ERDs for training
        test_erd = train_data2[i]
        actual_grade = train_grades2[i]
        
        # Create training set without current ERD
        train_erds_loo = train_data2[:i] + train_data2[i+1:]
        train_grades_loo = train_grades2[:i] + train_grades2[i+1:]
        
        # Predict
        predicted_grade = knn_predict(test_erd, train_erds_loo, train_grades_loo, k=K, threshold=q_threshold)
        
        predictions2.append(predicted_grade)
        actuals2.append(actual_grade)
    
    # Calculate RMSE
    predictions1 = np.array(predictions1)
    actuals1 = np.array(actuals1)
    predictions2 = np.array(predictions2)
    actuals2 = np.array(actuals2)
    
    rmse1 = np.sqrt(np.mean((predictions1 - actuals1) ** 2))
    rmse2 = np.sqrt(np.mean((predictions2 - actuals2) ** 2))
    
    # Combined RMSE
    all_predictions = np.concatenate([predictions1, predictions2])
    all_actuals = np.concatenate([actuals1, actuals2])
    rmse_combined = np.sqrt(np.mean((all_predictions - all_actuals) ** 2))
    
    print("\n" + "="*60)
    print("TRAINING RMSE RESULTS (Leave-One-Out Cross-Validation)")
    print("="*60)
    print(f"Dataset1 RMSE: {rmse1:.4f}")
    print(f"Dataset2 RMSE: {rmse2:.4f}")
    print(f"Combined RMSE: {rmse_combined:.4f}")
    print("="*60)
    
    # Also compute MAE for reference
    mae1 = np.mean(np.abs(predictions1 - actuals1))
    mae2 = np.mean(np.abs(predictions2 - actuals2))
    mae_combined = np.mean(np.abs(all_predictions - all_actuals))
    
    print(f"\nDataset1 MAE: {mae1:.4f}")
    print(f"Dataset2 MAE: {mae2:.4f}")
    print(f"Combined MAE: {mae_combined:.4f}")
    
    return rmse1, rmse2, rmse_combined

if __name__ == '__main__':
    compute_training_rmse()

