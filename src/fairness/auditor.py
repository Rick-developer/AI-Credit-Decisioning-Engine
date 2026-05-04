import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any

from src.data.models import FairnessReport

class FairnessAuditor:
    """Audits credit decisioning models for bias across protected demographic classes.
    
    Implements:
    1. Demographic Parity (Approval Rate equity)
    2. Proxy Variable Detection (Correlations with protected classes)
    """
    
    def __init__(self, protected_classes: List[str] = None):
        # Default protected classes in our UAE BNPL dataset
        self.protected_classes = protected_classes or ['age_group', 'gender', 'nationality']

    def check_demographic_parity(self, df_results: pd.DataFrame, threshold: float = 0.45) -> List[FairnessReport]:
        """Check if approval rates are equitable across protected groups.
        
        Args:
            df_results: DataFrame containing both the protected classes and the model's risk_score
            threshold: The risk score threshold below which applicants are approved
        """
        reports = []
        
        # Calculate approvals
        df_results['approved'] = (df_results['risk_score'] <= threshold).astype(int)
        overall_approval_rate = df_results['approved'].mean()
        
        for protected_attr in self.protected_classes:
            if protected_attr not in df_results.columns:
                continue
                
            # Calculate approval rate by group
            group_rates = df_results.groupby(protected_attr)['approved'].mean().to_dict()
            
            # Find max disparity
            rates = list(group_rates.values())
            if not rates:
                continue
                
            max_rate = max(rates)
            min_rate = min(rates)
            max_disparity = max_rate - min_rate
            
            # CBUAE/general industry threshold is often around 0.10 (10 percentage points)
            # or the 4/5ths rule (min_rate / max_rate >= 0.8)
            passes_threshold = max_disparity <= 0.15  # Using 15% for synthetic data leniency
            
            narrative = (
                f"Demographic Parity Audit for {protected_attr}: "
                f"{'PASSED' if passes_threshold else 'FAILED'}. "
                f"Overall approval rate is {overall_approval_rate:.1%}. "
                f"Maximum disparity between groups is {max_disparity:.1%}. "
            )
            
            reports.append(FairnessReport(
                metric_name="Demographic Parity",
                protected_attribute=protected_attr,
                group_results=group_rates,
                max_disparity=max_disparity,
                passes_threshold=passes_threshold,
                narrative=narrative
            ))
            
        return reports

    def detect_proxy_variables(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify features that are highly correlated with protected classes.
        
        Even if protected classes are dropped (Fairness Through Unawareness),
        the model can still learn bias through these proxy variables.
        """
        proxies = []
        
        # Features to check against protected classes
        features_to_check = [col for col in df.columns 
                           if pd.api.types.is_numeric_dtype(df[col]) 
                           and col not in self.protected_classes
                           and col not in ['applicant_id', 'defaulted', 'risk_score', 'approved']]
                           
        for protected_attr in self.protected_classes:
            if protected_attr not in df.columns:
                continue
                
            # Calculate Cramer's V or ANOVA depending on variable types
            # For simplicity in this portfolio project, we'll calculate
            # the variance of means across groups (ANOVA style)
            
            for feature in features_to_check:
                # Group feature by protected class
                groups = [group[feature].values for name, group in df.groupby(protected_attr) if len(group) > 10]
                
                if len(groups) < 2:
                    continue
                    
                # Perform One-way ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                
                # If p-value is very small, there's a strong relationship
                # We also check the effect size (eta squared approx)
                if p_value < 0.001:
                    # Calculate max difference in means as a proxy for effect size
                    means = df.groupby(protected_attr)[feature].mean()
                    max_diff_pct = (means.max() - means.min()) / max(abs(means.mean()), 0.001)
                    
                    if max_diff_pct > 0.2: # 20% difference in means
                        proxies.append({
                            "protected_class": protected_attr,
                            "proxy_feature": feature,
                            "p_value": p_value,
                            "max_difference_pct": max_diff_pct,
                            "warning": f"Feature '{feature}' may act as a proxy for '{protected_attr}' (Max mean difference: {max_diff_pct:.1%})"
                        })
                        
        # Sort by effect size
        proxies.sort(key=lambda x: x["max_difference_pct"], reverse=True)
        return proxies
