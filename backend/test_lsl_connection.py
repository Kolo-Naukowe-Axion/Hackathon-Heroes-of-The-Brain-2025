#!/usr/bin/env python3
"""
LSL Connection Diagnostic Tool
Run this to diagnose why the BCI is stuck in simulation mode.
"""

import time
from pylsl import resolve_streams, resolve_byprop, StreamInlet

def main():
    print("="*80)
    print("LSL CONNECTION DIAGNOSTIC TOOL")
    print("="*80)
    print("\nThis tool will help diagnose why the BCI is in simulation mode.")
    print("Make sure BrainAccess Board is running with LSL streaming enabled!\n")
    
    # Step 1: Check for any LSL streams
    print("STEP 1: Scanning for ALL LSL streams (timeout: 10 seconds)...")
    print("-" * 80)
    all_streams = resolve_streams(wait_time=10.0)
    
    if not all_streams:
        print("❌ RESULT: No LSL streams found in network!")
        print("\nPOSSIBLE CAUSES:")
        print("  1. BrainAccess Board application is not running")
        print("  2. Device is not connected to BrainAccess Board")
        print("  3. LSL streaming is not enabled in BrainAccess Board")
        print("  4. Firewall is blocking LSL (UDP broadcast on port 16571)")
        print("  5. BrainAccess Board and this script are on different networks")
        print("\nTROUBLESHOOTING STEPS:")
        print("  1. Open BrainAccess Board in your browser")
        print("  2. Connect your BrainAccess device")
        print("  3. Go to Stream/LSL tab")
        print("  4. Enable LSL streaming")
        print("  5. Check that the status shows 'Streaming'")
        print("  6. Run this script again")
        return
    
    print(f"✓ Found {len(all_streams)} LSL stream(s):\n")
    for i, stream in enumerate(all_streams):
        print(f"  Stream #{i+1}:")
        print(f"    Name:        '{stream.name()}'")
        print(f"    Type:        '{stream.type()}'")
        print(f"    Source ID:   '{stream.source_id()}'")
        print(f"    Hostname:    '{stream.hostname()}'")
        print(f"    Channels:    {stream.channel_count()}")
        print(f"    Sample Rate: {stream.nominal_srate()} Hz")
        print()
    
    # Step 2: Check specifically for EEG streams
    print("STEP 2: Searching for streams with type='EEG'...")
    print("-" * 80)
    eeg_streams = resolve_byprop('type', 'EEG', timeout=5.0)
    
    if eeg_streams:
        print(f"✓ Found {len(eeg_streams)} stream(s) with type='EEG':")
        for i, stream in enumerate(eeg_streams):
            print(f"  Stream: '{stream.name()}' (type: '{stream.type()}')")
        print("\n✅ SUCCESS: Backend should be able to connect!")
    else:
        print("❌ No streams found with type='EEG'")
        print("\nChecking if any streams might be EEG streams...")
        
        potential_eeg = []
        for stream in all_streams:
            stream_name = stream.name().lower()
            stream_type = stream.type().lower()
            is_potential = False
            reasons = []
            
            if 'brainaccess' in stream_name:
                is_potential = True
                reasons.append("name contains 'brainaccess'")
            if 'eeg' in stream_type or 'eeg' in stream_name:
                is_potential = True
                reasons.append("name/type contains 'eeg'")
            if stream.channel_count() >= 4:
                is_potential = True
                reasons.append(f"has {stream.channel_count()} channels (EEG typically has 4+)")
            
            if is_potential:
                potential_eeg.append((stream, reasons))
        
        if potential_eeg:
            print(f"\n⚠️  Found {len(potential_eeg)} potential EEG stream(s):")
            for stream, reasons in potential_eeg:
                print(f"\n  Stream: '{stream.name()}' (type: '{stream.type()}')")
                print(f"    Reasons: {', '.join(reasons)}")
                print(f"    Channels: {stream.channel_count()}")
            
            print("\n⚠️  WARNING: These streams don't have type='EEG'")
            print("   The backend requires type='EEG' to auto-detect the stream.")
            print("   You may need to configure BrainAccess Board to set stream type to 'EEG'")
        else:
            print("❌ No potential EEG streams found.")
    
    # Step 3: Test connection to first stream
    if all_streams:
        print("\nSTEP 3: Testing connection to first stream...")
        print("-" * 80)
        test_stream = all_streams[0]
        print(f"Testing stream: '{test_stream.name()}'")
        
        try:
            inlet = StreamInlet(test_stream)
            print("✓ StreamInlet created successfully")
            
            print("Attempting to pull data (timeout: 5 seconds)...")
            chunk, timestamps = inlet.pull_chunk(timeout=5.0)
            
            if chunk and len(chunk) > 0:
                print(f"✓ SUCCESS: Received {len(chunk)} samples!")
                print(f"  Sample shape: {len(chunk)} x {len(chunk[0]) if chunk else 0}")
                print(f"  First sample values: {chunk[0] if chunk else 'N/A'}")
                print("\n✅ Stream is working! Data is being received.")
            else:
                print("⚠️  WARNING: Stream exists but no data received")
                print("   This might mean:")
                print("   - Stream just started (wait a few seconds)")
                print("   - Stream is paused")
                print("   - Connection issue")
        except Exception as e:
            print(f"❌ ERROR connecting to stream: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if not all_streams:
        print("1. Start BrainAccess Board application")
        print("2. Connect your BrainAccess device")
        print("3. Enable LSL streaming in BrainAccess Board")
        print("4. Check firewall settings (allow UDP port 16571)")
    elif not eeg_streams:
        print("1. Configure BrainAccess Board to set stream type to 'EEG'")
        print("2. Check BrainAccess Board settings for LSL stream configuration")
        print("3. The backend code can be modified to accept other stream types")
    else:
        print("✅ Everything looks good! The backend should connect automatically.")
        print("   If it's still in simulation mode, check the backend logs for errors.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

