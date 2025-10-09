# -*- coding: utf-8 -*-
"""
Enhanced AML Detection System
Combines rule-based scoring with multiple ML models (Logistic Regression, Random Forest, Isolation Forest)
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, 
                           roc_curve, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV
import joblib
from typing import Dict, List, Tuple, Optional
import json

# Import our custom modules
from currency_normalizer import CurrencyNormalizer
from file_validator import FileValidator

class EnhancedAMLDetector:
    """
    Enhanced AML detection system combining multiple approaches
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the enhanced AML detector
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Initialize components
        self.currency_normalizer = CurrencyNormalizer()
        self.file_validator = FileValidator()
        
        # Model storage
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters"""
        return {
            # Data processing
            'high_value_threshold': 20_000,
            'odd_hours_start': 0,
            'odd_hours_end': 4,
            'structuring_window_hours': 24,
            'velocity_window_hours': 24,
            'diversity_window_days': 1,
            'structuring_min_count': 5,
            'velocity_min_count': 20,
            'counterparty_diversity_min': 10,
            
            # Model parameters
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            
            # Random Forest parameters
            'rf_n_estimators': 100,
            'rf_max_depth': 10,
            'rf_min_samples_split': 5,
            'rf_min_samples_leaf': 2,
            
            # Isolation Forest parameters
            'if_n_estimators': 200,
            'if_contamination': 0.1,
            'if_max_samples': 0.8,
            
            # Ensemble weights
            'ensemble_weights': {
                'logistic_regression': 0.3,
                'random_forest': 0.4,
                'isolation_forest': 0.3
            },
            
            # Output settings
            'save_models': True,
            'make_plots': True,
            'output_dir': 'outputs'
        }
    
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess transaction data with validation and currency normalization
        
        Args:
            file_path: Path to transaction data file
            
        Returns:
            Preprocessed DataFrame
        """
        print("=" * 60)
        print("LOADING AND PREPROCESSING DATA")
        print("=" * 60)
        
        # Validate file first
        print("Validating file...")
        is_valid, report, validation_result = self.file_validator.validate_uploaded_file(file_path)
        
        if not is_valid:
            print("❌ File validation failed!")
            print(report)
            raise ValueError("File validation failed. Please check the file and try again.")
        
        print("✅ File validation passed")
        print(f"File info: {validation_result['file_info']['name']} "
              f"({validation_result['file_info']['size_mb']:.2f} MB)")
        
        # Load data
        print("Loading data...")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        print(f"Loaded {len(df)} transactions with {len(df.columns)} columns")
        
        # Normalize column names
        df = self._normalize_columns(df)
        
        # Normalize currencies to USD
        print("Normalizing currencies to USD...")
        df = self.currency_normalizer.normalize_to_usd(df)
        
        # Generate rule-based features
        print("Generating rule-based features...")
        df = self._generate_rule_features(df)
        
        # Generate additional ML features
        print("Generating ML features...")
        df = self._generate_ml_features(df)
        
        print(f"Final dataset shape: {df.shape}")
        print("✅ Data preprocessing completed")
        
        return df
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and ensure required columns exist"""
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # Map common column variations
        name_map = {
            "Timestamp": "timestamp",
            "From Bank": "from_bank",
            "To Bank": "to_bank",
            "Amount Received": "amount_received",
            "Receiving Currency": "receiving_currency",
            "Amount Paid": "amount_paid",
            "Payment Currency": "payment_currency",
            "Payment Format": "payment_format",
            "Is Laundering": "is_laundering",
        }
        
        for old_name, new_name in name_map.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Map account columns
        cols = df.columns.tolist()
        acct_like = [i for i, c in enumerate(cols) if c.replace(".", "").lower().startswith("account")]
        if len(acct_like) >= 2:
            cols[acct_like[0]] = "from_account"
            cols[acct_like[1]] = "to_account"
            df.columns = cols
        
        # Ensure required columns exist
        required_cols = [
            "timestamp", "from_bank", "to_bank", "from_account", "to_account",
            "amount_received", "amount_paid", "payment_currency", "receiving_currency",
            "payment_format"
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan if col in ["amount_received", "amount_paid"] else ""
        
        # Parse timestamps and amounts
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        for amt in ["amount_received", "amount_paid"]:
            if amt in df.columns:
                s = df[amt].astype(str).str.replace(",", "", regex=False).str.strip()
                df[amt] = pd.to_numeric(s.replace({"": np.nan}), errors="coerce").astype("float32")
        
        return df
    
    def _generate_rule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rule-based features"""
        # Choose canonical amount column
        amount_col = "amount_paid_usd" if "amount_paid_usd" in df.columns else "amount_paid"
        if amount_col not in df.columns:
            amount_col = "amount_received_usd" if "amount_received_usd" in df.columns else "amount_received"
        
        amt = df[amount_col].fillna(0.0).astype("float32")
        
        # Basic rule flags
        df["flag_high_value"] = (amt > self.config['high_value_threshold'])
        df["flag_cross_bank"] = (df["from_bank"].astype(str).values != df["to_bank"].astype(str).values)
        df["flag_same_account"] = (df["from_account"].astype(str).values == df["to_account"].astype(str).values)
        
        # Time-based features
        df["hour"] = df["timestamp"].dt.hour
        df["flag_odd_hours"] = df["hour"].between(
            self.config['odd_hours_start'], 
            self.config['odd_hours_end'], 
            inclusive="both"
        )
        
        # Currency features
        if {"payment_currency", "receiving_currency"}.issubset(df.columns):
            df["flag_cross_currency"] = (
                df["payment_currency"].astype(str).str.upper().values
                != df["receiving_currency"].astype(str).str.upper().values
            )
        else:
            df["flag_cross_currency"] = False
        
        # Round amount detection
        df["flag_round_amount"] = (
            np.isclose((amt % 1000), 0) | np.isclose((amt * 100) % 100, 0)
        ).astype(bool)
        
        # Time-window patterns
        df = self._generate_time_window_features(df, amt)
        
        return df
    
    def _generate_time_window_features(self, df: pd.DataFrame, amt: pd.Series) -> pd.DataFrame:
        """Generate time-window based features"""
        # Time buckets
        df["ts_hour"] = df["timestamp"].dt.floor("H")
        df["ts_day"] = df["timestamp"].dt.floor("D")
        
        have_time = df["ts_hour"].notna().any()
        if not have_time:
            df["flag_structuring"] = False
            df["flag_velocity_outgoing"] = False
            df["flag_counterparty_diversity"] = False
            return df
        
        # Structuring detection
        small = amt < self.config['high_value_threshold']
        small_df = df.loc[small, ["from_account", "to_account", "ts_hour"]].dropna(subset=["ts_hour"])
        
        if not small_df.empty:
            hourly_ft = (
                small_df.groupby(["from_account", "to_account", "ts_hour"], sort=True)
                .size().rename("cnt").reset_index()
            )
            
            hourly_ft = hourly_ft.sort_values(["from_account", "to_account", "ts_hour"]).set_index("ts_hour")
            rolling_counts = (
                hourly_ft.groupby(["from_account", "to_account"])["cnt"]
                .rolling(f"{self.config['structuring_window_hours']}H")
                .sum()
                .reset_index()
            )
            rolling_counts["struct_hit"] = rolling_counts["cnt"] >= self.config['structuring_min_count']
            
            df = df.merge(
                rolling_counts[["from_account", "to_account", "ts_hour", "struct_hit"]],
                on=["from_account", "to_account", "ts_hour"],
                how="left"
            )
            df["flag_structuring"] = df["struct_hit"].fillna(False).values
            df.drop(columns=["struct_hit"], inplace=True)
        else:
            df["flag_structuring"] = False
        
        # Velocity detection
        hourly_from = (
            df.dropna(subset=["ts_hour"])
            .groupby(["from_account", "ts_hour"], sort=True)
            .size().rename("out_cnt").reset_index()
        )
        
        if not hourly_from.empty:
            hourly_from = hourly_from.sort_values(["from_account", "ts_hour"]).set_index("ts_hour")
            vel_roll = (
                hourly_from.groupby(["from_account"])["out_cnt"]
                .rolling(f"{self.config['velocity_window_hours']}H").sum()
                .reset_index()
            )
            vel_roll["vel_hit"] = vel_roll["out_cnt"] >= self.config['velocity_min_count']
            
            df = df.merge(
                vel_roll[["from_account", "ts_hour", "vel_hit"]],
                on=["from_account", "ts_hour"],
                how="left"
            )
            df["flag_velocity_outgoing"] = df["vel_hit"].fillna(False).values
            df.drop(columns=["vel_hit"], inplace=True)
        else:
            df["flag_velocity_outgoing"] = False
        
        # Counterparty diversity
        daily_div = (
            df.dropna(subset=["ts_day"])
            .groupby(["from_account", "ts_day"])["to_account"]
            .nunique()
            .rename("uniq_to")
            .reset_index()
        )
        
        if not daily_div.empty:
            daily_div["div_hit"] = daily_div["uniq_to"] >= self.config['counterparty_diversity_min']
            
            df = df.merge(
                daily_div[["from_account", "ts_day", "div_hit"]],
                on=["from_account", "ts_day"],
                how="left"
            )
            df["flag_counterparty_diversity"] = df["div_hit"].fillna(False).values
            df.drop(columns=["div_hit"], inplace=True)
        else:
            df["flag_counterparty_diversity"] = False
        
        return df
    
    def _generate_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate additional features for ML models"""
        # Amount-based features
        amount_col = "amount_paid_usd" if "amount_paid_usd" in df.columns else "amount_paid"
        if amount_col not in df.columns:
            amount_col = "amount_received_usd" if "amount_received_usd" in df.columns else "amount_received"
        
        amt = df[amount_col].fillna(0.0)
        
        # Statistical features
        df["amount_log"] = np.log1p(amt)
        df["amount_sqrt"] = np.sqrt(amt)
        df["amount_rank"] = amt.rank(pct=True)
        
        # Time features
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6])
        df["month"] = df["timestamp"].dt.month
        df["quarter"] = df["timestamp"].dt.quarter
        
        # Account activity features
        df["from_account_freq"] = df.groupby("from_account")["from_account"].transform("count")
        df["to_account_freq"] = df.groupby("to_account")["to_account"].transform("count")
        
        # Bank features
        df["from_bank_freq"] = df.groupby("from_bank")["from_bank"].transform("count")
        df["to_bank_freq"] = df.groupby("to_bank")["to_bank"].transform("count")
        
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare data for ML models"""
        # Select features
        rule_features = [
            "flag_high_value", "flag_cross_bank", "flag_same_account",
            "flag_odd_hours", "flag_cross_currency", "flag_round_amount",
            "flag_structuring", "flag_velocity_outgoing", "flag_counterparty_diversity"
        ]
        
        ml_features = [
            "amount_log", "amount_sqrt", "amount_rank", "day_of_week", 
            "is_weekend", "month", "quarter", "from_account_freq", 
            "to_account_freq", "from_bank_freq", "to_bank_freq"
        ]
        
        # Categorical features
        categorical_features = ["from_bank", "to_bank", "payment_currency", "receiving_currency", "payment_format"]
        
        # Combine all features
        all_features = rule_features + ml_features + categorical_features
        
        # Filter available features
        available_features = [f for f in all_features if f in df.columns]
        
        # Prepare feature matrix
        X = df[available_features].copy()
        
        # Encode categorical variables
        for col in categorical_features:
            if col in X.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    X[col] = self.encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(0)
        
        # Get target variable
        if "is_laundering" in df.columns:
            y = df["is_laundering"].astype(int)
        else:
            # Create dummy target for unsupervised learning
            y = pd.Series([0] * len(df), index=df.index)
        
        return X, y, available_features
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Dict:
        """Train all ML models"""
        print("=" * 60)
        print("TRAINING ML MODELS")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], 
            stratify=y if y.nunique() > 1 else None
        )
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        results = {}
        
        # 1. Logistic Regression (Rule-based weighting)
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=self.config['random_state'])
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        
        # Evaluate LR
        lr_pred = lr_model.predict(X_test)
        lr_proba = lr_model.predict_proba(X_test)[:, 1]
        
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred, zero_division=0),
            'recall': recall_score(y_test, lr_pred, zero_division=0),
            'f1': f1_score(y_test, lr_pred, zero_division=0),
            'auc': roc_auc_score(y_test, lr_proba) if y_test.nunique() > 1 else 0
        }
        
        # Feature importance for LR
        self.feature_importance['logistic_regression'] = dict(zip(feature_names, lr_model.coef_[0]))
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=self.config['rf_n_estimators'],
            max_depth=self.config['rf_max_depth'],
            min_samples_split=self.config['rf_min_samples_split'],
            min_samples_leaf=self.config['rf_min_samples_leaf'],
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # Evaluate RF
        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
            'f1': f1_score(y_test, rf_pred, zero_division=0),
            'auc': roc_auc_score(y_test, rf_proba) if y_test.nunique() > 1 else 0
        }
        
        # Feature importance for RF
        self.feature_importance['random_forest'] = dict(zip(feature_names, rf_model.feature_importances_))
        
        # 3. Isolation Forest (for anomaly detection)
        print("Training Isolation Forest...")
        # Use contamination rate based on actual fraud rate
        contamination_rate = max(0.01, min(0.5, y.mean())) if y.nunique() > 1 else 0.1
        
        if_model = IsolationForest(
            n_estimators=self.config['if_n_estimators'],
            contamination=contamination_rate,
            max_samples=self.config['if_max_samples'],
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        if_model.fit(X_train_scaled)
        self.models['isolation_forest'] = if_model
        
        # Evaluate IF
        if_pred = if_model.predict(X_test_scaled)
        if_scores = if_model.score_samples(X_test_scaled)
        
        # Convert to binary predictions (1 for anomaly, 0 for normal)
        if_pred_binary = (if_pred == -1).astype(int)
        
        results['isolation_forest'] = {
            'accuracy': accuracy_score(y_test, if_pred_binary),
            'precision': precision_score(y_test, if_pred_binary, zero_division=0),
            'recall': recall_score(y_test, if_pred_binary, zero_division=0),
            'f1': f1_score(y_test, if_pred_binary, zero_division=0),
            'auc': roc_auc_score(y_test, -if_scores) if y_test.nunique() > 1 else 0
        }
        
        # Store performance metrics
        self.model_performance = results
        
        # Print results
        print("\nModel Performance Summary:")
        print("-" * 50)
        for model_name, metrics in results.items():
            print(f"{model_name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  AUC:       {metrics['auc']:.4f}")
            print()
        
        return results
    
    def create_ensemble_prediction(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create ensemble predictions from all models"""
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        if 'logistic_regression' in self.models:
            lr_pred = self.models['logistic_regression'].predict(X)
            lr_proba = self.models['logistic_regression'].predict_proba(X)[:, 1]
            predictions['logistic_regression'] = lr_pred
            probabilities['logistic_regression'] = lr_proba
        
        if 'random_forest' in self.models:
            rf_pred = self.models['random_forest'].predict(X)
            rf_proba = self.models['random_forest'].predict_proba(X)[:, 1]
            predictions['random_forest'] = rf_pred
            probabilities['random_forest'] = rf_proba
        
        if 'isolation_forest' in self.models:
            X_scaled = self.scalers['standard'].transform(X)
            if_pred = self.models['isolation_forest'].predict(X_scaled)
            if_scores = self.models['isolation_forest'].score_samples(X_scaled)
            # Convert to probability-like scores
            if_proba = 1 / (1 + np.exp(if_scores))  # Sigmoid transformation
            predictions['isolation_forest'] = (if_pred == -1).astype(int)
            probabilities['isolation_forest'] = if_proba
        
        # Weighted ensemble
        weights = self.config['ensemble_weights']
        ensemble_proba = np.zeros(len(X))
        ensemble_pred = np.zeros(len(X))
        
        for model_name, proba in probabilities.items():
            if model_name in weights:
                ensemble_proba += weights[model_name] * proba
        
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        return ensemble_pred, ensemble_proba
    
    def generate_final_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate final risk scores and severity levels"""
        print("=" * 60)
        print("GENERATING FINAL SCORES")
        print("=" * 60)
        
        # Prepare ML data
        X, y, feature_names = self.prepare_ml_data(df)
        
        # Get ensemble predictions
        ensemble_pred, ensemble_proba = self.create_ensemble_prediction(X)
        
        # Add scores to dataframe
        df['ensemble_risk_score'] = ensemble_proba
        df['ensemble_prediction'] = ensemble_pred
        
        # Create severity levels
        # Use quantiles to create balanced severity distribution
        risk_quantiles = df['ensemble_risk_score'].quantile([0.7, 0.85, 0.95])
        
        def assign_severity(score):
            if score >= risk_quantiles[0.95]:
                return 'Critical'
            elif score >= risk_quantiles[0.85]:
                return 'High'
            elif score >= risk_quantiles[0.7]:
                return 'Medium'
            else:
                return 'Low'
        
        df['severity'] = df['ensemble_risk_score'].apply(assign_severity)
        
        # Add individual model scores for explainability
        if 'logistic_regression' in self.models:
            lr_proba = self.models['logistic_regression'].predict_proba(X)[:, 1]
            df['lr_risk_score'] = lr_proba
        
        if 'random_forest' in self.models:
            rf_proba = self.models['random_forest'].predict_proba(X)[:, 1]
            df['rf_risk_score'] = rf_proba
        
        if 'isolation_forest' in self.models:
            X_scaled = self.scalers['standard'].transform(X)
            if_scores = self.models['isolation_forest'].score_samples(X_scaled)
            df['if_anomaly_score'] = -if_scores  # Higher scores = more anomalous
        
        return df
    
    def save_models(self):
        """Save trained models and preprocessors"""
        if not self.config['save_models']:
            return
        
        print("Saving models and preprocessors...")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
        
        # Save scalers and encoders
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(self.model_dir, f"scaler_{name}.joblib")
            joblib.dump(scaler, scaler_path)
        
        for name, encoder in self.encoders.items():
            encoder_path = os.path.join(self.model_dir, f"encoder_{name}.joblib")
            joblib.dump(encoder, encoder_path)
        
        # Save configuration and feature importance
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        importance_path = os.path.join(self.model_dir, "feature_importance.json")
        with open(importance_path, 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        print(f"✅ Models saved to {self.model_dir}")
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualization plots"""
        if not self.config['make_plots']:
            return
        
        print("Creating visualizations...")
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 1. Risk score distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(df['ensemble_risk_score'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Ensemble Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        
        # 2. Severity distribution
        plt.subplot(2, 2, 2)
        severity_counts = df['severity'].value_counts()
        plt.bar(severity_counts.index, severity_counts.values)
        plt.title('Transaction Severity Distribution')
        plt.xlabel('Severity Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 3. Feature importance (Random Forest)
        if 'random_forest' in self.feature_importance:
            plt.subplot(2, 2, 3)
            rf_importance = self.feature_importance['random_forest']
            top_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importance = zip(*top_features)
            plt.barh(range(len(features)), importance)
            plt.yticks(range(len(features)), features)
            plt.title('Top 10 Random Forest Feature Importance')
            plt.xlabel('Importance')
        
        # 4. Model comparison
        if self.model_performance:
            plt.subplot(2, 2, 4)
            models = list(self.model_performance.keys())
            f1_scores = [self.model_performance[model]['f1'] for model in models]
            plt.bar(models, f1_scores)
            plt.title('Model F1-Score Comparison')
            plt.xlabel('Model')
            plt.ylabel('F1-Score')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'aml_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to {self.config['output_dir']}")
    
    def run_full_analysis(self, file_path: str) -> pd.DataFrame:
        """Run the complete AML analysis pipeline"""
        start_time = time.time()
        
        print("🚀 Starting Enhanced AML Detection Analysis")
        print("=" * 60)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(file_path)
        
        # Prepare ML data
        X, y, feature_names = self.prepare_ml_data(df)
        
        # Train models
        if y.nunique() > 1:  # Only train if we have labels
            self.train_models(X, y, feature_names)
        else:
            print("No labels available, skipping supervised model training")
        
        # Generate final scores
        df = self.generate_final_scores(df)
        
        # Save models
        self.save_models()
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Save results
        output_path = os.path.join(self.config['output_dir'], 'enhanced_aml_results.csv')
        os.makedirs(self.config['output_dir'], exist_ok=True)
        df.to_csv(output_path, index=False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        severity_summary = df['severity'].value_counts()
        print("Severity Distribution:")
        for severity, count in severity_summary.items():
            print(f"  {severity}: {count:,} transactions ({count/len(df)*100:.1f}%)")
        
        print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
        print(f"Results saved to: {output_path}")
        
        return df


if __name__ == "__main__":
    # Example usage
    detector = EnhancedAMLDetector()
    
    # Run analysis on a transaction file
    # df = detector.run_full_analysis("HI-Medium_Trans.csv")
    
    print("Enhanced AML Detector initialized successfully!")
    print("Use detector.run_full_analysis('your_file.csv') to analyze transaction data")
