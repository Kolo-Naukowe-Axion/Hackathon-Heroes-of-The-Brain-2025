# BCI Pipeline Issues - Fixed

## Summary
This document outlines all the issues found in the BCI (Brain-Computer Interface) pipeline and the fixes applied.

## Issues Found and Fixed

### 1. **Duplicate Code Lines** ✅ FIXED
- **Location**: `backend/brainaccess_live.py`
- **Issue**: 
  - Line 87-88: `self.current_probs` was assigned twice
  - Line 134-137: `"emotion"` and `"probabilities"` were duplicated in the return dictionary
- **Fix**: Removed duplicate lines

### 2. **Model Architecture Mismatch** ✅ FIXED
- **Location**: `backend/brainaccess_live.py`
- **Issue**: 
  - Code was using `ResNetMLP` architecture but the saved model weights use `WideResNet` architecture (from `model_v4/inference.py`)
  - Model weights have structure `net.0.weight` but code expected `input_layer.0.weight`
- **Fix**: 
  - Replaced `ResNetMLP` with `WideResNet` class definition matching `model_v4/inference.py`
  - Updated model loading to use correct architecture

### 3. **Missing Ensemble Model Support** ✅ FIXED
- **Location**: `backend/brainaccess_live.py`
- **Issue**: 
  - The `ml_components.pkl` contains a `gb_model` (Gradient Boosting) for ensemble predictions
  - Code was only using ResNet model, ignoring the ensemble
  - Model_v4 uses 60% ResNet + 40% GB ensemble
- **Fix**: 
  - Added `gb_model` loading from components
  - Implemented ensemble prediction: `0.6 * p_res + 0.4 * p_gb`

### 4. **Missing Feature Preprocessing** ✅ FIXED
- **Location**: `backend/brainaccess_live.py`
- **Issue**: 
  - Model_v4 expects `np.log1p(X_raw)` before polynomial transformation
  - Code was applying polynomial transform directly on features
- **Fix**: Added `np.log1p()` preprocessing step before polynomial transformation

### 5. **Emotion Mapping Mismatch** ✅ FIXED
- **Location**: `backend/brainaccess_live.py` and `frontend/src/App.jsx`
- **Issue**: 
  - Model outputs: `['Boring', 'Calm', 'Horror', 'Funny']` (4 classes)
  - Backend was mapping to: `{0: "neutral", 1: "happy", 2: "sad", 3: "angry"}` (wrong!)
  - Frontend expects: `neutral, calm, happy, sad, angry` (5 emotions)
- **Fix**: 
  - Created `MODEL_TO_FRONTEND_MAP` to correctly map model outputs:
    - `Boring` → `neutral`
    - `Calm` → `calm`
    - `Horror` → `angry`
    - `Funny` → `happy`
  - Updated mock loop to include all 5 frontend emotions

### 6. **Confidence Threshold Mismatch** ✅ FIXED
- **Location**: `backend/brainaccess_live.py`
- **Issue**: 
  - Code used `CONFIDENCE_THRESHOLD = 0.4` but model_v4 uses `0.65`
- **Fix**: Updated threshold to `0.65` to match model_v4

### 7. **Missing Channel Count Validation** ✅ FIXED
- **Location**: `backend/brainaccess_live.py`
- **Issue**: 
  - Code handled case where channels > 4 but didn't properly handle channels < 4
- **Fix**: Added proper validation and warning message for insufficient channels

### 8. **Mock Mode Emotion List** ✅ FIXED
- **Location**: `backend/brainaccess_live.py`
- **Issue**: 
  - Mock mode only cycled through 4 emotions, missing "calm"
- **Fix**: Updated mock emotions to include all 5 frontend emotions with proper probability mapping

## Testing Recommendations

1. **Test with actual BrainAccess device**:
   - Verify LSL stream connection
   - Check that 4 channels are received correctly
   - Validate emotion predictions match expected outputs

2. **Test mock mode**:
   - Verify all 5 emotions cycle correctly
   - Check WebSocket data format matches frontend expectations

3. **Test ensemble model**:
   - Verify `gb_model` loads correctly
   - Check that ensemble predictions are different from ResNet-only predictions

4. **Test frontend integration**:
   - Verify emotion labels map correctly from backend to frontend
   - Check that all 5 emotion states are reachable

## Files Modified

- `backend/brainaccess_live.py` - Complete refactor to fix all issues

## Notes

- The model architecture now matches `model_v4/inference.py` exactly
- All preprocessing steps now match the training pipeline
- Emotion mapping is now consistent between model outputs and frontend expectations
- The code is more robust with better error handling for edge cases

