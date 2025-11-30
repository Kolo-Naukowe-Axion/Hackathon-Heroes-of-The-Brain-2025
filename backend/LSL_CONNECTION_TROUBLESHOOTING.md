# LSL Connection Troubleshooting Guide

## Problem: BCI is Constantly in Simulation Mode

If your BCI is stuck in simulation mode, it means the system cannot find the LSL stream from BrainAccess Board.

---

## Quick Diagnostic

Run the diagnostic tool:
```bash
python test_lsl_connection.py
```

This will tell you:
- If any LSL streams are found
- If streams have the correct type='EEG'
- If data can be received from the stream

---

## Common Causes & Solutions

### 1. BrainAccess Board Not Running
**Symptom**: No LSL streams found at all

**Solution**:
- Open BrainAccess Board in your web browser
- Make sure it's running and accessible

---

### 2. Device Not Connected
**Symptom**: BrainAccess Board shows "Disconnected"

**Solution**:
- Connect your BrainAccess device (Halo/Cap)
- Wait for status to show "Connected"
- Check device battery level

---

### 3. LSL Streaming Not Enabled
**Symptom**: Device connected but no LSL stream

**Solution**:
1. In BrainAccess Board, go to **Stream** or **LSL** tab
2. Find the **LSL Stream** toggle or **Enable LSL** button
3. **Turn it ON**
4. Verify it shows "Streaming" status

---

### 4. Stream Type Not Set to 'EEG'
**Symptom**: Streams found but backend doesn't recognize them

**Solution**:
- Check BrainAccess Board LSL settings
- Ensure stream type is set to **'EEG'** (case-sensitive)
- If you can't change it, the backend will try to auto-detect by name

---

### 5. Firewall Blocking LSL
**Symptom**: No streams found even when everything else is correct

**Solution**:
- LSL uses UDP broadcast on port **16571**
- Check Windows Firewall settings
- Allow Python/pylsl through firewall
- Check if antivirus is blocking network traffic

---

### 6. Network Issues
**Symptom**: Streams found but can't connect

**Solution**:
- Make sure BrainAccess Board and backend are on same network
- If using VPN, try disabling it
- Check network adapter settings

---

## Improved Detection Logic

The backend now has improved stream detection:

1. **Primary Search**: Looks for streams with `type='EEG'`
2. **Fallback Search**: If not found, searches for:
   - Streams with 'brainaccess' in name
   - Streams with 'eeg' in name or type
   - Streams with 4+ channels (typical for EEG)

3. **Extended Timeout**: Increased from 2-3 seconds to 10 seconds for initial scan

4. **Better Logging**: Shows detailed information about found streams

---

## Manual Testing Steps

1. **Start BrainAccess Board**
   ```
   - Open in browser
   - Connect device
   - Enable LSL streaming
   ```

2. **Run Diagnostic Tool**
   ```bash
   python test_lsl_connection.py
   ```
   This will show you exactly what streams are available.

3. **Check Backend Logs**
   Look for these messages:
   - `"Scanning for LSL streams..."`
   - `"Found X LSL stream(s)"`
   - `"✓ Found stream: ..."`
   - `"✓✓✓ is_mock = False (REAL CONNECTION MODE)"`

4. **If Still in Simulation**
   - Check the console output for error messages
   - Verify stream type is 'EEG' in BrainAccess Board
   - Try restarting both BrainAccess Board and backend

---

## What the Backend Does

### Initial Connection Attempt
1. Scans for LSL streams (10 second timeout)
2. Lists all found streams with details
3. Searches for type='EEG' streams
4. Falls back to name-based detection if needed
5. If found: connects and verifies data reception
6. If not found: switches to simulation mode

### Simulation Mode Behavior
- Generates mock emotion data
- **Automatically checks for device every 10 seconds**
- Will switch to real mode when device is found
- Shows reconnect check messages in console

---

## Expected Console Output

### When Stream is Found:
```
Scanning for LSL streams (this may take up to 10 seconds)...
✓ Found 1 LSL stream(s) in network:
  Stream 1:
    Name: 'BrainAccess'
    Type: 'EEG'
    Channels: 4
    Sampling Rate: 250.0 Hz

✓ Found stream: BrainAccess (type: EEG)
✓✓✓ is_mock = False (REAL CONNECTION MODE) ✓✓✓
Connected to BrainAccess and receiving data!
```

### When Stream is NOT Found:
```
Scanning for LSL streams...
✗ No LSL streams found in network.

⚠️ No EEG stream found! Switching to SIMULATION mode.
RUNNING IN SIMULATION MODE
[RECONNECT CHECK #1] Checking for LSL stream...
  → No LSL streams found.
```

---

## Still Having Issues?

1. **Run the diagnostic tool** and share the output
2. **Check BrainAccess Board** - is LSL actually enabled?
3. **Check firewall** - is UDP port 16571 blocked?
4. **Try restarting** both BrainAccess Board and backend
5. **Check backend logs** for specific error messages

---

## Code Changes Made

The following improvements were made to help with connection:

1. **Extended timeout**: 10 seconds instead of 2-3 seconds
2. **Better logging**: Shows all stream details
3. **Flexible detection**: Accepts streams with 4+ channels even if type isn't 'EEG'
4. **Improved reconnection**: Better messages and more frequent checks in simulation mode
5. **Diagnostic tool**: New `test_lsl_connection.py` script for troubleshooting

