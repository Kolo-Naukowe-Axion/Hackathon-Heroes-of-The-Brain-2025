# Pipeline Analysis Report

## Data Flow Pipeline (Excluding Model)

### 1. LSL Stream Reception ✅
- **Location**: `_run_loop()` line 271
- **Method**: `inlet.pull_chunk(timeout=1.0)`
- **Status**: OK - Non-blocking with timeout
- **Issue Found**: None
- **Recommendation**: Keep as-is

### 2. Buffer Management ⚠️
- **Location**: Line 255 - `deque(maxlen=BUFFER_LENGTH)`
- **Issue**: Buffer uses `maxlen=250` which means old samples are automatically dropped
- **Potential Problem**: If samples arrive slower than expected, buffer might never fill
- **Status**: Need to verify sample rate matches expectation (250Hz)
- **Recommendation**: Add buffer fill rate monitoring

### 3. Prediction Timing 🔍
- **Location**: Lines 346-353
- **Issue**: Predictions only happen when:
  1. Buffer is full (250 samples)
  2. Time since last prediction >= 0.1s
- **Potential Problem**: If chunks are small or arrive infrequently, predictions might be delayed
- **Status**: Logic is correct but may be too restrictive
- **Recommendation**: Allow predictions with nearly-full buffer (already done at line 348)

### 4. Feature Extraction ✅
- **Location**: `extract_band_powers()` lines 69-87
- **Status**: OK - Pure function, no side effects
- **Issue Found**: None
- **Note**: Uses Welch's method, computationally stable

### 5. State Update (CRITICAL) ⚠️⚠️
- **Location**: Lines 501-521
- **Issue**: State update happens INSIDE prediction block
- **Problem**: If prediction doesn't run, state never updates!
- **Status**: This is a major issue - state only updates when predictions happen
- **Recommendation**: State should update more frequently or WebSocket should read buffer state directly

### 6. Thread Safety ✅
- **Location**: Line 501 - `with self.lock:`
- **Status**: OK - Proper locking for state updates
- **Issue Found**: None

### 7. WebSocket Data Retrieval ⚠️
- **Location**: `server.py` line 45 - `detector.get_data()`
- **Issue**: Calls `get_data()` every 0.1s but if predictions don't run, returns stale data
- **Problem**: WebSocket rate (10Hz) doesn't match prediction rate (10Hz when buffer full)
- **Status**: Rate mismatch can cause stale data
- **Recommendation**: Either reduce WebSocket rate or ensure state updates happen more frequently

### 8. Frontend Reception ✅
- **Location**: `App.jsx` lines 88-116
- **Status**: OK - Properly parses and updates state
- **Issue Found**: None

## CRITICAL ISSUES FOUND

### Issue #1: State Only Updates During Predictions ⚠️⚠️⚠️
**Problem**: `self.current_probs` and `self.current_emotion` are ONLY updated inside the prediction block (line 505-506). If predictions don't happen (buffer not full, timing, etc.), the WebSocket keeps sending stale data.

**Impact**: Even if new data arrives, if predictions don't trigger, frontend sees constant values.

**Fix Needed**: State should be updated even when predictions don't run, OR predictions should be guaranteed to run continuously.

### Issue #2: Buffer Fill Rate Assumption ⚠️
**Problem**: Code assumes 250Hz sample rate. If actual rate differs, buffer may fill slower/faster.

**Impact**: Predictions might happen too rarely or too frequently.

**Fix Needed**: Add dynamic buffer size or sample rate detection.

### Issue #3: WebSocket Sends Stale Data ⚠️
**Problem**: `get_data()` returns whatever is in `self.current_probs` which might be from last prediction (could be seconds ago).

**Impact**: Frontend receives old probabilities even when new data arrives.

**Fix Needed**: Either timestamp the data or ensure state updates happen independently of predictions.

## RECOMMENDATIONS

1. **Add timestamp to state**: Track when state was last updated
2. **Separate data ingestion from prediction**: Update state with raw buffer info even without prediction
3. **Add buffer health monitoring**: Log buffer fill rate and prediction frequency
4. **Make predictions more frequent**: Reduce buffer requirement or prediction interval
5. **Add data freshness check**: WebSocket should know if data is stale

