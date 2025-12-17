"""
Unit tests for Fraud Detection API.

This module contains tests for validating the fraud detection API
endpoints and prediction logic.

Author: Arvind Kumar
Version: 1.0.0
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestFraudAPI:
    """Test cases for Fraud Detection API."""

    def test_risk_level_critical(self):
        """Test critical risk level for high probability."""
        from src.fraud_api import get_risk_level
        level, _ = get_risk_level(0.95)
        assert level == "CRITICAL"

    def test_risk_level_high(self):
        """Test high risk level."""
        from src.fraud_api import get_risk_level
        level, _ = get_risk_level(0.75)
        assert level == "HIGH"

    def test_risk_level_medium(self):
        """Test medium risk level."""
        from src.fraud_api import get_risk_level
        level, _ = get_risk_level(0.55)
        assert level == "MEDIUM"

    def test_risk_level_low(self):
        """Test low risk level."""
        from src.fraud_api import get_risk_level
        level, _ = get_risk_level(0.35)
        assert level == "LOW"

    def test_risk_level_minimal(self):
        """Test minimal risk level."""
        from src.fraud_api import get_risk_level
        level, _ = get_risk_level(0.1)
        assert level == "MINIMAL"


class TestFeatureValidation:
    """Test cases for feature validation."""

    def test_feature_count(self):
        """Verify correct number of features."""
        expected_features = [
            'amount', 'hour', 'day_of_week', 'merchant_category',
            'distance_from_home', 'distance_from_last_transaction',
            'ratio_to_median_purchase', 'repeat_retailer',
            'used_chip', 'used_pin', 'online_order'
        ]
        assert len(expected_features) == 11
