import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score
from typing import Any, Tuple, Dict, List

from src.data.models import CreditDecision
from src.features.engineer import engineer_features

class CreditRiskModel:
    """XGBoost model for credit risk assessment with SHAP explanations.
    
    Demonstrates 'Fairness Through Unawareness' by explicitly dropping 
    protected demographic classes before training, setting up the 
    Fairness Auditor (Layer 3) to prove that this doesn't prevent proxy bias.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=4,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.explainer = None
        self.feature_names = None
        
        # Protected classes that MUST NOT be used for training
        self.protected_features = ['age_group', 'gender', 'nationality']
        # IDs and targets that shouldn't be in features
        self.meta_features = ['applicant_id', 'defaulted', 'requires_bureau_check']
        
        self.categorical_features = ['employment_type']

    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Engineer features and drop protected/meta columns."""
        # 1. Engineer domain features
        df_eng = engineer_features(df)
        
        # 2. One-hot encode categoricals
        df_encoded = pd.get_dummies(df_eng, columns=self.categorical_features, drop_first=True)
        
        # 3. Drop protected classes and meta columns
        cols_to_drop = [c for c in self.protected_features + self.meta_features if c in df_encoded.columns]
        X = df_encoded.drop(columns=cols_to_drop)
        
        if is_training:
            self.feature_names = list(X.columns)
        
        # Ensure all columns are numeric
        cols = list(X.columns)
        for col in cols:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col])
                except Exception:
                    X = X.drop(columns=[col])
                    if is_training and col in self.feature_names:
                        self.feature_names.remove(col)
                        
        return X

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the model and return evaluation metrics."""
        X = self._prepare_features(df, is_training=True)
        y = df['defaulted']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model.fit(X_train, y_train)
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Evaluate
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        metrics = {
            "auc_roc": float(roc_auc_score(y_test, y_prob)),
            "brier_score": float(brier_score_loss(y_test, y_prob)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0))
        }
        return metrics

    def predict_with_explanation(self, df_applicant: pd.DataFrame, threshold: float = 0.45) -> CreditDecision:
        """Predict risk for a single applicant and generate SHAP attributions."""
        if len(df_applicant) != 1:
            raise ValueError("Expected exactly 1 applicant record")
            
        applicant_id = df_applicant['applicant_id'].iloc[0]
        
        X = self._prepare_features(df_applicant)
        
        # Make sure columns match training
        if self.feature_names:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
            
        # Predict
        prob_default = float(self.model.predict_proba(X)[0, 1])
        approved = prob_default <= threshold
        
        # SHAP explanation
        shap_values = self.explainer.shap_values(X)[0]
        
        # Get top 3 risk-increasing factors (positive SHAP values)
        # and top 2 risk-decreasing factors (negative SHAP values)
        feature_impacts = [{"feature": feat, "impact": float(val), "value": float(X[feat].iloc[0])} 
                           for feat, val in zip(X.columns, shap_values)]
        
        # Sort by absolute impact for the top overall factors
        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_factors = feature_impacts[:5]
        
        return CreditDecision(
            applicant_id=applicant_id,
            approved=approved,
            risk_score=prob_default,
            threshold_used=threshold,
            top_factors=top_factors,
            adverse_action_notice=None,
            reason_codes=[],
            regulatory_citations=[]
        )

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions on a batch of applicants for fairness testing."""
        X = self._prepare_features(df)
        probs = self.model.predict_proba(X)[:, 1]
        
        result_df = df.copy()
        result_df['risk_score'] = probs
        return result_df
