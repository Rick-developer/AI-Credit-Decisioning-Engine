"""
grounder.py — Policy grounding engine for credit decisions (Layer 4).

Given a CreditDecision (with SHAP factors and reason codes), this module
retrieves the relevant CBUAE regulatory provisions and populates the
`regulatory_citations` field.

Retrieval strategy:
1. Extract feature names from the decision's top_factors
2. Extract reason codes
3. Match against the keyword index of the curated regulatory corpus
4. Score by keyword overlap (TF-style: more keyword matches = more relevant)
5. Return top-K unique regulations, formatted as citations

This is NOT a vector-similarity search. It is a deterministic, auditable
keyword retrieval — appropriate for a bounded regulatory surface area.
For full semantic RAG, see the UAE Regulatory Compliance RAG Agent project.
"""

from typing import List, Dict, Any
from collections import Counter

from src.data.models import CreditDecision
from src.policy.regulations import CBUAE_CORPUS, Regulation


class PolicyGrounder:
    """Grounds credit decisions in CBUAE regulatory provisions.
    
    Layer 4 of the hybrid ML+LLM+Fairness+Policy architecture.
    Takes a CreditDecision and returns it enriched with regulatory citations.
    """
    
    def __init__(self, corpus: List[Regulation] = None):
        """Initialize with a regulatory corpus.
        
        Args:
            corpus: List of Regulation objects. Defaults to CBUAE_CORPUS.
        """
        self.corpus = corpus or CBUAE_CORPUS
        self._build_keyword_index()
    
    def _build_keyword_index(self) -> None:
        """Build an inverted index: keyword → list of regulation IDs."""
        self._keyword_to_regs: Dict[str, List[str]] = {}
        self._reg_lookup: Dict[str, Regulation] = {}
        
        for reg in self.corpus:
            self._reg_lookup[reg.id] = reg
            for keyword in reg.keywords:
                kw_lower = keyword.lower()
                if kw_lower not in self._keyword_to_regs:
                    self._keyword_to_regs[kw_lower] = []
                self._keyword_to_regs[kw_lower].append(reg.id)
    
    def _extract_signals(self, decision: CreditDecision) -> List[str]:
        """Extract all searchable signals from a credit decision.
        
        Combines:
        - Feature names from SHAP top_factors
        - Reason codes (lowercased)
        - Decision state signals (approved/declined)
        """
        signals = []
        
        # 1. Feature names from SHAP
        for factor in decision.top_factors:
            feat = factor.get("feature", "")
            if feat:
                signals.append(feat.lower())
        
        # 2. Reason codes
        for code in decision.reason_codes:
            signals.append(code.lower())
        
        # 3. Decision state
        if not decision.approved:
            signals.extend(["declined", "denied", "adverse_action_notice"])
        
        # 4. Threshold context
        if decision.risk_score > 0.7:
            signals.append("high_risk")
        
        return signals
    
    def retrieve(self, decision: CreditDecision, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve the most relevant regulatory provisions for a decision.
        
        Args:
            decision: The CreditDecision to ground.
            top_k: Maximum number of citations to return.
            
        Returns:
            List of dicts with 'id', 'source', 'provision' keys.
        """
        signals = self._extract_signals(decision)
        
        # Score each regulation by keyword overlap
        reg_scores: Counter = Counter()
        
        for signal in signals:
            # Exact match
            if signal in self._keyword_to_regs:
                for reg_id in self._keyword_to_regs[signal]:
                    reg_scores[reg_id] += 1
            
            # Partial match — check if the signal is a substring of any keyword
            for keyword, reg_ids in self._keyword_to_regs.items():
                if signal in keyword or keyword in signal:
                    for reg_id in reg_ids:
                        reg_scores[reg_id] += 0.5
        
        if not reg_scores:
            return []
        
        # Return top-K, sorted by score descending
        top_reg_ids = [reg_id for reg_id, _ in reg_scores.most_common(top_k)]
        
        citations = []
        for reg_id in top_reg_ids:
            reg = self._reg_lookup[reg_id]
            citations.append({
                "id": reg.id,
                "source": reg.source,
                "provision": reg.provision,
            })
        
        return citations
    
    def ground_decision(self, decision: CreditDecision, top_k: int = 3) -> CreditDecision:
        """Enrich a CreditDecision with regulatory citations.
        
        This is the main entry point for Layer 4. It modifies the decision
        in-place and also returns it for chaining.
        
        Args:
            decision: The CreditDecision to ground.
            top_k: Maximum number of citations to include.
            
        Returns:
            The same CreditDecision with `regulatory_citations` populated.
        """
        citations = self.retrieve(decision, top_k=top_k)
        
        # Format citations as structured strings for the CreditDecision model
        decision.regulatory_citations = [
            f"[{c['id']}] {c['source']}: {c['provision']}"
            for c in citations
        ]
        
        return decision
