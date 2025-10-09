# -*- coding: utf-8 -*-
"""
Explainability Engine for AML Detection
Provides SHAP and LIME explanations for transaction flagging decisions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")

import joblib
import os
from datetime import datetime
import json

class AMLExplainabilityEngine:
    """
    Explainability engine for AML detection models
    """
    
    def __init__(self, model_dir: str = "models", output_dir: str = "explanations"):
        """
        Initialize the explainability engine
        
        Args:
            model_dir: Directory containing trained models
            output_dir: Directory to save explanation outputs
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.explainers = {}
        
        # Load models and preprocessors
        self._load_models()
    
    def _load_models(self):
        """Load trained models and preprocessors"""
        try:
            # Load models
            model_files = {
                'logistic_regression': 'logistic_regression.joblib',
                'random_forest': 'random_forest.joblib',
                'isolation_forest': 'isolation_forest.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.model_dir, filename)
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} model")
            
            # Load scalers
            scaler_path = os.path.join(self.model_dir, 'scaler_standard.joblib')
            if os.path.exists(scaler_path):
                self.scalers['standard'] = joblib.load(scaler_path)
            
            # Load encoders
            for encoder_file in os.listdir(self.model_dir):
                if encoder_file.startswith('encoder_') and encoder_file.endswith('.joblib'):
                    encoder_name = encoder_file.replace('encoder_', '').replace('.joblib', '')
                    encoder_path = os.path.join(self.model_dir, encoder_file)
                    self.encoders[encoder_name] = joblib.load(encoder_path)
            
            # Load feature names from config
            config_path = os.path.join(self.model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # We'll need to reconstruct feature names from the data
            
            print(f"Loaded {len(self.models)} models and {len(self.encoders)} encoders")
            
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
    
    def prepare_explanation_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for explanation (same preprocessing as training)
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Tuple of (processed_features, feature_names)
        """
        # Rule features
        rule_features = [
            "flag_high_value", "flag_cross_bank", "flag_same_account",
            "flag_odd_hours", "flag_cross_currency", "flag_round_amount",
            "flag_structuring", "flag_velocity_outgoing", "flag_counterparty_diversity"
        ]
        
        # ML features
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
        
        # Encode categorical variables using loaded encoders
        for col in categorical_features:
            if col in X.columns and col in self.encoders:
                try:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
                except:
                    # Handle unseen categories
                    X[col] = X[col].astype(str).map(
                        lambda x: 0 if x not in self.encoders[col].classes_ else x
                    )
                    X[col] = self.encoders[col].transform(X[col])
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, available_features
    
    def create_shap_explainers(self, X: pd.DataFrame, feature_names: List[str]):
        """Create SHAP explainers for available models"""
        if not SHAP_AVAILABLE:
            print("SHAP not available, skipping SHAP explanations")
            return
        
        print("Creating SHAP explainers...")
        
        # Sample data for explainer training (SHAP can be slow on large datasets)
        sample_size = min(1000, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'logistic_regression':
                    # Linear explainer for logistic regression
                    explainer = shap.LinearExplainer(model, X_sample)
                elif model_name == 'random_forest':
                    # Tree explainer for random forest
                    explainer = shap.TreeExplainer(model)
                elif model_name == 'isolation_forest':
                    # Use Kernel explainer for isolation forest
                    def if_predict(X_input):
                        if 'standard' in self.scalers:
                            X_scaled = self.scalers['standard'].transform(X_input)
                        else:
                            X_scaled = X_input
                        return -model.score_samples(X_scaled)  # Convert to anomaly scores
                    
                    explainer = shap.KernelExplainer(if_predict, X_sample)
                else:
                    continue
                
                self.explainers[f"shap_{model_name}"] = explainer
                print(f"Created SHAP explainer for {model_name}")
                
            except Exception as e:
                print(f"Could not create SHAP explainer for {model_name}: {e}")
    
    def create_lime_explainers(self, X: pd.DataFrame, feature_names: List[str]):
        """Create LIME explainers for available models"""
        if not LIME_AVAILABLE:
            print("LIME not available, skipping LIME explanations")
            return
        
        print("Creating LIME explainers...")
        
        # Convert to numpy array for LIME
        X_array = X.values
        
        for model_name, model in self.models.items():
            try:
                # Create prediction function
                if model_name == 'isolation_forest':
                    def predict_fn(X_input):
                        if 'standard' in self.scalers:
                            X_scaled = self.scalers['standard'].transform(X_input)
                        else:
                            X_scaled = X_input
                        scores = -model.score_samples(X_scaled)
                        # Convert to probabilities
                        return np.column_stack([1 - scores, scores])
                else:
                    def predict_fn(X_input):
                        return model.predict_proba(X_input)
                
                # Create LIME explainer
                explainer = LimeTabularExplainer(
                    X_array,
                    feature_names=feature_names,
                    class_names=['Normal', 'Suspicious'],
                    mode='classification'
                )
                
                self.explainers[f"lime_{model_name}"] = (explainer, predict_fn)
                print(f"Created LIME explainer for {model_name}")
                
            except Exception as e:
                print(f"Could not create LIME explainer for {model_name}: {e}")
    
    def explain_transaction(self, transaction_idx: int, X: pd.DataFrame, 
                          feature_names: List[str], df_original: pd.DataFrame) -> Dict:
        """
        Generate explanations for a specific transaction
        
        Args:
            transaction_idx: Index of the transaction to explain
            X: Feature matrix
            feature_names: List of feature names
            df_original: Original DataFrame with transaction details
            
        Returns:
            Dictionary containing all explanations
        """
        explanations = {
            'transaction_info': {},
            'shap_explanations': {},
            'lime_explanations': {},
            'rule_explanations': {},
            'summary': {}
        }
        
        # Get transaction data
        transaction_data = X.iloc[transaction_idx:transaction_idx+1]
        original_transaction = df_original.iloc[transaction_idx]
        
        # Store transaction info
        explanations['transaction_info'] = {
            'index': transaction_idx,
            'timestamp': str(original_transaction.get('timestamp', 'Unknown')),
            'from_account': str(original_transaction.get('from_account', 'Unknown')),
            'to_account': str(original_transaction.get('to_account', 'Unknown')),
            'amount': float(original_transaction.get('amount_paid_usd', original_transaction.get('amount_paid', 0))),
            'severity': str(original_transaction.get('severity', 'Unknown')),
            'risk_score': float(original_transaction.get('ensemble_risk_score', 0))
        }
        
        # Generate SHAP explanations
        if SHAP_AVAILABLE:
            for explainer_name, explainer in self.explainers.items():
                if explainer_name.startswith('shap_'):
                    model_name = explainer_name.replace('shap_', '')
                    try:
                        shap_values = explainer.shap_values(transaction_data)
                        
                        # Handle different SHAP output formats
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]  # Use positive class
                        
                        explanations['shap_explanations'][model_name] = {
                            'values': shap_values[0].tolist(),
                            'feature_names': feature_names,
                            'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0
                        }
                        
                    except Exception as e:
                        print(f"SHAP explanation failed for {model_name}: {e}")
        
        # Generate LIME explanations
        if LIME_AVAILABLE:
            for explainer_name, (explainer, predict_fn) in self.explainers.items():
                if explainer_name.startswith('lime_'):
                    model_name = explainer_name.replace('lime_', '')
                    try:
                        lime_exp = explainer.explain_instance(
                            transaction_data.values[0], 
                            predict_fn, 
                            num_features=min(10, len(feature_names))
                        )
                        
                        explanations['lime_explanations'][model_name] = {
                            'explanation': lime_exp.as_list(),
                            'score': lime_exp.score
                        }
                        
                    except Exception as e:
                        print(f"LIME explanation failed for {model_name}: {e}")
        
        # Generate rule-based explanations
        explanations['rule_explanations'] = self._explain_rules(original_transaction)
        
        # Generate summary
        explanations['summary'] = self._generate_explanation_summary(explanations)
        
        return explanations
    
    def _explain_rules(self, transaction: pd.Series) -> Dict:
        """Generate rule-based explanations"""
        rule_explanations = {}
        
        # High value flag
        if transaction.get('flag_high_value', False):
            amount = transaction.get('amount_paid_usd', transaction.get('amount_paid', 0))
            rule_explanations['high_value'] = {
                'triggered': True,
                'reason': f"Transaction amount (${amount:,.2f}) exceeds high-value threshold",
                'impact': 'High'
            }
        
        # Cross-bank flag
        if transaction.get('flag_cross_bank', False):
            from_bank = transaction.get('from_bank', 'Unknown')
            to_bank = transaction.get('to_bank', 'Unknown')
            rule_explanations['cross_bank'] = {
                'triggered': True,
                'reason': f"Cross-bank transaction: {from_bank} → {to_bank}",
                'impact': 'Medium'
            }
        
        # Same account flag
        if transaction.get('flag_same_account', False):
            account = transaction.get('from_account', 'Unknown')
            rule_explanations['same_account'] = {
                'triggered': True,
                'reason': f"Self-transfer detected: {account} → {account}",
                'impact': 'Medium'
            }
        
        # Odd hours flag
        if transaction.get('flag_odd_hours', False):
            hour = transaction.get('hour', 'Unknown')
            rule_explanations['odd_hours'] = {
                'triggered': True,
                'reason': f"Transaction occurred during odd hours: {hour}:00",
                'impact': 'Low'
            }
        
        # Cross-currency flag
        if transaction.get('flag_cross_currency', False):
            from_curr = transaction.get('payment_currency', 'Unknown')
            to_curr = transaction.get('receiving_currency', 'Unknown')
            rule_explanations['cross_currency'] = {
                'triggered': True,
                'reason': f"Cross-currency transaction: {from_curr} → {to_curr}",
                'impact': 'Medium'
            }
        
        # Round amount flag
        if transaction.get('flag_round_amount', False):
            amount = transaction.get('amount_paid_usd', transaction.get('amount_paid', 0))
            rule_explanations['round_amount'] = {
                'triggered': True,
                'reason': f"Round amount detected: ${amount:,.2f}",
                'impact': 'Low'
            }
        
        # Structuring flag
        if transaction.get('flag_structuring', False):
            rule_explanations['structuring'] = {
                'triggered': True,
                'reason': "Multiple small transactions to same counterparty detected",
                'impact': 'High'
            }
        
        # Velocity flag
        if transaction.get('flag_velocity_outgoing', False):
            rule_explanations['velocity'] = {
                'triggered': True,
                'reason': "High velocity of outgoing transactions detected",
                'impact': 'High'
            }
        
        # Counterparty diversity flag
        if transaction.get('flag_counterparty_diversity', False):
            rule_explanations['counterparty_diversity'] = {
                'triggered': True,
                'reason': "High number of distinct counterparties detected",
                'impact': 'Medium'
            }
        
        return rule_explanations
    
    def _generate_explanation_summary(self, explanations: Dict) -> Dict:
        """Generate a summary of all explanations"""
        summary = {
            'risk_factors': [],
            'key_insights': [],
            'recommendation': 'Review transaction manually'
        }
        
        # Collect risk factors from rules
        rule_explanations = explanations.get('rule_explanations', {})
        for rule_name, rule_info in rule_explanations.items():
            if rule_info['triggered']:
                summary['risk_factors'].append({
                    'factor': rule_name.replace('_', ' ').title(),
                    'impact': rule_info['impact'],
                    'reason': rule_info['reason']
                })
        
        # Sort by impact
        impact_order = {'High': 3, 'Medium': 2, 'Low': 1}
        summary['risk_factors'].sort(key=lambda x: impact_order.get(x['impact'], 0), reverse=True)
        
        # Generate key insights
        if summary['risk_factors']:
            high_impact_factors = [f for f in summary['risk_factors'] if f['impact'] == 'High']
            if high_impact_factors:
                summary['key_insights'].append(f"High-risk factors detected: {len(high_impact_factors)}")
            
            medium_impact_factors = [f for f in summary['risk_factors'] if f['impact'] == 'Medium']
            if medium_impact_factors:
                summary['key_insights'].append(f"Medium-risk factors detected: {len(medium_impact_factors)}")
        
        # Generate recommendation
        risk_score = explanations['transaction_info'].get('risk_score', 0)
        if risk_score > 0.8:
            summary['recommendation'] = 'Immediate investigation required'
        elif risk_score > 0.6:
            summary['recommendation'] = 'Enhanced due diligence recommended'
        elif risk_score > 0.4:
            summary['recommendation'] = 'Monitor for similar patterns'
        else:
            summary['recommendation'] = 'Low risk - routine processing'
        
        return summary
    
    def explain_high_risk_transactions(self, df: pd.DataFrame, top_n: int = 10) -> Dict:
        """
        Generate explanations for the highest risk transactions
        
        Args:
            df: DataFrame with transaction data and risk scores
            top_n: Number of top transactions to explain
            
        Returns:
            Dictionary containing explanations for top transactions
        """
        print(f"Generating explanations for top {top_n} high-risk transactions...")
        
        # Prepare data
        X, feature_names = self.prepare_explanation_data(df)
        
        # Create explainers
        self.create_shap_explainers(X, feature_names)
        self.create_lime_explainers(X, feature_names)
        
        # Get top risky transactions
        if 'ensemble_risk_score' in df.columns:
            top_risky = df.nlargest(top_n, 'ensemble_risk_score')
        else:
            print("No risk scores found, using random sample")
            top_risky = df.sample(n=min(top_n, len(df)))
        
        explanations = {}
        
        for idx, (_, transaction) in enumerate(top_risky.iterrows()):
            transaction_idx = transaction.name
            print(f"Explaining transaction {idx + 1}/{top_n} (index: {transaction_idx})")
            
            try:
                explanation = self.explain_transaction(transaction_idx, X, feature_names, df)
                explanations[f"transaction_{transaction_idx}"] = explanation
            except Exception as e:
                print(f"Failed to explain transaction {transaction_idx}: {e}")
        
        return explanations
    
    def save_explanations(self, explanations: Dict, filename: str = None):
        """Save explanations to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aml_explanations_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(explanations, f, indent=2, default=str)
        
        print(f"Explanations saved to: {filepath}")
        return filepath
    
    def create_explanation_report(self, explanations: Dict, output_file: str = None) -> str:
        """Create a human-readable explanation report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"aml_explanation_report_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, output_file)
        
        with open(filepath, 'w') as f:
            f.write("AML TRANSACTION EXPLANATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            for trans_key, explanation in explanations.items():
                if not trans_key.startswith('transaction_'):
                    continue
                
                # Transaction info
                info = explanation['transaction_info']
                f.write(f"TRANSACTION: {trans_key}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Timestamp: {info['timestamp']}\n")
                f.write(f"From: {info['from_account']} → To: {info['to_account']}\n")
                f.write(f"Amount: ${info['amount']:,.2f}\n")
                f.write(f"Risk Score: {info['risk_score']:.4f}\n")
                f.write(f"Severity: {info['severity']}\n\n")
                
                # Risk factors
                risk_factors = explanation['summary']['risk_factors']
                if risk_factors:
                    f.write("RISK FACTORS:\n")
                    for factor in risk_factors:
                        f.write(f"  • {factor['factor']} ({factor['impact']} impact)\n")
                        f.write(f"    Reason: {factor['reason']}\n")
                    f.write("\n")
                
                # Key insights
                insights = explanation['summary']['key_insights']
                if insights:
                    f.write("KEY INSIGHTS:\n")
                    for insight in insights:
                        f.write(f"  • {insight}\n")
                    f.write("\n")
                
                # Recommendation
                f.write(f"RECOMMENDATION: {explanation['summary']['recommendation']}\n\n")
                f.write("=" * 60 + "\n\n")
        
        print(f"Explanation report saved to: {filepath}")
        return filepath


def explain_aml_decisions(df: pd.DataFrame, model_dir: str = "models", 
                         top_n: int = 10) -> Dict:
    """
    Convenience function to explain AML decisions
    
    Args:
        df: DataFrame with transaction data and risk scores
        model_dir: Directory containing trained models
        top_n: Number of top transactions to explain
        
    Returns:
        Dictionary containing explanations
    """
    engine = AMLExplainabilityEngine(model_dir=model_dir)
    explanations = engine.explain_high_risk_transactions(df, top_n=top_n)
    
    # Save explanations
    engine.save_explanations(explanations)
    engine.create_explanation_report(explanations)
    
    return explanations


if __name__ == "__main__":
    print("AML Explainability Engine")
    print("Use explain_aml_decisions(df) to generate explanations for transaction data")
