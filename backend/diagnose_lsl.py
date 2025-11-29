from pylsl import resolve_streams, resolve_byprop

print("--- LSL DIAGNOSTIC TOOL ---")
print("Attempting to resolve ALL LSL streams (timeout=5s)...")

try:
    streams = resolve_streams(wait_time=5.0)
    if not streams:
        print("RESULT: No LSL streams found.")
        print("POSSIBLE CAUSES:")
        print("1. BrainAccess Board application is not running.")
        print("2. Device is not connected/turned on.")
        print("3. Firewall is blocking LSL (UDP broadcast).")
    else:
        print(f"RESULT: Found {len(streams)} streams:")
        for i, stream in enumerate(streams):
            print(f"  Stream {i+1}:")
            print(f"    Name: {stream.name()}")
            print(f"    Type: {stream.type()}")
            print(f"    ID: {stream.source_id()}")
            print(f"    Hostname: {stream.hostname()}")
            
        # Check specifically for EEG
        print("\nChecking specifically for 'type=EEG'...")
        eeg_streams = resolve_byprop('type', 'EEG', timeout=2.0)
        if eeg_streams:
            print("  SUCCESS: Found 'EEG' stream(s). Backend should work.")
        else:
            print("  WARNING: Streams exist, but none have type='EEG'. Backend requires type='EEG'.")

except Exception as e:
    print(f"ERROR during resolution: {e}")
