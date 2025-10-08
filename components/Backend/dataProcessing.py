def randomForest(X_train_orig, y_train_orig, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    # Apply SMOTE only on the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_orig, y_train_orig)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train_resampled)

    # Predict probabilities on test set
    probs = rf.predict_proba(X_test_scaled)[:, 1]
    chosen_threshold = 0.5
    y_pred = (probs >= chosen_threshold).astype(int)

    return rf, chosen_threshold, X_test_scaled, y_test, y_pred

def generate_report(y_test, y_pred, df_original, test_indices):
    import os
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

    # Calculate metrics
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)

    total = len(y_test)
    pred_fraud = int((y_pred == 1).sum())
    pred_normal = int((y_pred == 0).sum())
    fraud_rate = pred_fraud / total if total else 0.0

    # Get the subset of original dataframe that corresponds to the test indices
    df_test = df_original.loc[test_indices].copy()
    df_test = df_test.reset_index(drop=True)
    df_test['predicted_class'] = y_pred

    # Sample flagged transactions (first 8)
    flagged_samples = df_test[df_test['predicted_class'] == 1].head(8)
    if flagged_samples.empty:
        examples_txt = "No flagged transactions."
    else:
        examples_txt = flagged_samples.to_string(index=False)

    # Prepare report text
    report_text = f"""
    Fraud Detection Report
    ----------------------
    Total transactions analyzed: {total}
    Predicted fraud count: {pred_fraud} ({fraud_rate:.2%})
    Predicted normal count: {pred_normal}

    Classification Metrics:
      - Precision: {prec:.4f}
      - Recall:    {rec:.4f}
      - F1-score:  {f1:.4f}
      - Confusion Matrix: {cm.tolist()}

    Sample flagged transactions (up to 8):
    {examples_txt}

    Recommendations:
    - Review flagged transactions manually for verification.
    - Adjust threshold based on business risk tolerance.
    - Retrain model periodically with updated data.
    - Monitor false positives and false negatives carefully.
    """

    print(report_text)


# Main script
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    dataTrans = "components/Backend/dataset/HI-Medium_Trans.csv"
    df_trans = pd.read_csv(dataTrans)

    df_model = df_trans.copy()
    df_model= df_model.sample(50000, random_state=42)

    # Drop irrelevant or unique ID columns
    colsToDrop = ['Timestamp', 'Account', 'Account.1']
    df_model.drop(columns=colsToDrop, inplace=True)

    # Encode categorical features
    for col in ['Receiving Currency', 'Payment Currency', 'Payment Format']:
        top_values = df_model[col].value_counts().nlargest(5).index
        df_model[col] = df_model[col].where(df_model[col].isin(top_values), 'Other')

    df_model = pd.get_dummies(df_model, columns=['Receiving Currency', 'Payment Currency', 'Payment Format'], drop_first=True)

    df_model.drop_duplicates(inplace=True)
    df_model.dropna(inplace=True)

    y = df_model['Is Laundering'].astype(int)
    X = df_model.drop(columns=['Is Laundering'])

    # Split data once and keep test indices
    X_train_orig, X_test_orig, y_train_orig, y_test_orig, train_idx, test_idx = train_test_split(
        X, y, X.index, test_size=0.3, stratify=y, random_state=42
    )

    # Run model
    rf, threshold, X_test_scaled, y_test, y_pred = randomForest(
        X_train_orig, y_train_orig, X_test_orig, y_test_orig
    )

    # Generate report
    generate_report(y_test, y_pred, df_model, test_idx)

except Exception as e:
    print("Error:", e)

