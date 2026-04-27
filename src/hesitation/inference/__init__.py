"""High-level inference interface for hesitation detection model.

This module provides a clean, deterministic API for MATLAB/external systems
to call the trained hesitation model.

Usage:
    from hesitation.inference import HesitationPredictor

    predictor = HesitationPredictor.load_default()
    prediction = predictor.predict_single(feature_dict)
    print(prediction)
"""

from hesitation.inference.predictor import HesitationPredictor, Prediction

__all__ = ["HesitationPredictor", "Prediction"]
