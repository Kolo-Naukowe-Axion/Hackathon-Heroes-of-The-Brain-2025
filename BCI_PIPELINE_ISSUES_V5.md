# BCI Pipeline Issues - Model v5 Migration & Fixes

## Summary
This document outlines all the issues found and fixed when migrating to model_v5 and reviewing the BCI pipeline implementation.

## Critical Issues Found and Fixed

### 1. **Double Log Transform Bug** ✅ FIXED
- **Location**: `backend/brainaccess_live.py` - `extract_band_powers()` function
- **Issue**: 
  - Function was applying `np.log10(band_power)` on line 77
  - Then later applying `np.log1p(features)` on line 189
  - This resulted in a double log transform: `log1p(log10(power))` which is incorrect
  - Model_v5 expects raw band powers, then applies `log1p` in preprocessing
- **Fix**: 
  - Changed `extract_band_powers()` to return raw band powers (removed `np.log10`)
  - Added documentation explaining that model_v5 applies log1p in preprocessing
  - Preprocessing pipeline now matches model_v5 exactly: `raw_powers → log1p → poly → scaler`

### 2. **Device Handling Not Implemented** ✅ FIXED
- **Location**: `backend/brainaccess_live.py - EmotionDetector.__init__()`
- **Issue**: 
  - Model was always loaded to CPU with `map_location=torch.device('cpu')`
  - Model was never moved to GPU even if available
  - Model_v5/inference.py properly uses GPU when available
- **Fix**: 
  - Added device detection: `self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
  - Model is now moved to the appropriate device: `self.model = self.model.to(self.device)`
  - Input tensors are moved to device: `input_tensor.to(self.device)`
  - Results are moved back to CPU for numpy conversion: `.cpu().numpy()`

### 3. **Missing Error Handling for gb_model** ✅ FIXED
- **Location**: `backend/brainaccess_live.py` - Model loading and prediction
- **Issue**: 
  - Code checked if `gb_model` exists but didn't warn if missing
  - Model_v5/inference.py assumes `gb_model` always exists (line 67), which could crash
  - No fallback behavior documented
- **Fix**: 
  - Added warning message when `gb_model` is not found
  - Added fallback to ResNet-only predictions when `gb_model` is None
  - Added comment explaining the ensemble weights (60% ResNet, 40% GB)

### 4. **Unused Import** ✅ FIXED
- **Location**: `backend/brainaccess_live.py` - Line 5
- **Issue**: `pandas` was imported but never used
- **Fix**: Removed unused `import pandas as pd`

### 5. **Missing Model Validation** ✅ FIXED
- **Location**: `backend/brainaccess_live.py` - `_run_loop()` method
- **Issue**: 
  - If model loading failed, `self.model` would be None
  - Code would crash when trying to use `self.model(input_tensor)` on line 213
  - No error handling for this edge case
- **Fix**: 
  - Added check at the start of prediction loop: `if self.model is None:`
  - Added error message and continue to skip prediction if model not loaded
  - Prevents crashes and provides clear error feedback

### 6. **Preprocessing Pipeline Verification** ✅ VERIFIED
- **Location**: `backend/brainaccess_live.py` - Feature preprocessing
- **Status**: 
  - Verified preprocessing matches model_v5 exactly:
    1. Extract raw band powers (20 features: 4 channels × 5 bands)
    2. Apply `log1p` transform
    3. Apply polynomial transformation (20 → 230 features)
    4. Apply scaler normalization
    5. Run inference
  - Matches model_v5/inference.py line 60: `scaler.transform(poly.transform(log1p(X_raw)))`

## Additional Improvements Made

### 7. **Better Error Messages**
- Added device information in initialization logs
- Added warning for missing gb_model
- Added error message when model is None during prediction

### 8. **Code Documentation**
- Added docstring to `extract_band_powers()` explaining the return format
- Added comments explaining preprocessing steps
- Added comments explaining ensemble weights

## Testing Recommendations

1. **Test preprocessing pipeline**:
   - Verify raw band powers are extracted correctly
   - Check that log1p transform is applied correctly
   - Validate polynomial transformation produces 230 features
   - Confirm scaler normalization works

2. **Test device handling**:
   - Test on system with GPU available
   - Test on system with CPU only
   - Verify model runs on correct device

3. **Test ensemble model**:
   - Verify `gb_model` loads from model_v5 components
   - Test fallback behavior if `gb_model` is missing
   - Validate ensemble weights (60/40 split)

4. **Test error handling**:
   - Test behavior when model files are missing
   - Test behavior when model fails to load
   - Verify graceful degradation

5. **Test with actual BrainAccess device**:
   - Verify 4-channel EEG stream processing
   - Check band power extraction
   - Validate emotion predictions

## Files Modified

- `backend/brainaccess_live.py` - Fixed all issues listed above

## Notes

- The preprocessing pipeline now exactly matches model_v5/inference.py
- Device handling now matches model_v5 implementation
- Error handling is more robust with better user feedback
- Code is cleaner with unused imports removed
- All fixes maintain backward compatibility with existing functionality

