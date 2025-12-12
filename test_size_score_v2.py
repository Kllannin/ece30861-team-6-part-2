#!/usr/bin/env python3
"""Test the updated size score calculation"""

from multiprocessing import Queue
from src.metrics.calculate_size_score import calculate_size_score

# Test cases
test_cases = [
    ("Unknown size (0 bytes)", 0, {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.5, "aws_server": 1.0}),
    ("Tiny model (50MB)", 50 * 1024 * 1024, {"raspberry_pi": 1.0, "jetson_nano": 1.0, "desktop_pc": 1.0, "aws_server": 1.0}),
    ("Small model (500MB)", 500 * 1024 * 1024, {"raspberry_pi": 0.5, "jetson_nano": 1.0, "desktop_pc": 1.0, "aws_server": 1.0}),
    ("Medium model (3GB)", 3 * 1024 * 1024 * 1024, {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 1.0, "aws_server": 1.0}),
    ("Large model (15GB)", 15 * 1024 * 1024 * 1024, {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.75, "aws_server": 1.0}),
    ("Very large model (75GB)", 75 * 1024 * 1024 * 1024, {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.25, "aws_server": 0.9}),
    ("Huge model (150GB)", 150 * 1024 * 1024 * 1024, {"raspberry_pi": 0.0, "jetson_nano": 0.0, "desktop_pc": 0.25, "aws_server": 0.75}),
]

print("Testing updated size score calculation:\n")
print("=" * 100)

log_queue = Queue()
passed = 0
failed = 0

for name, size_bytes, expected in test_cases:
    scores, latency = calculate_size_score(size_bytes, verbosity=0, log_queue=log_queue)
    
    # Check if all scores match expected
    all_match = all(scores.get(key) == expected.get(key) for key in expected)
    
    status = "✓ PASS" if all_match else "✗ FAIL"
    if all_match:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} | {name}")
    print(f"     Size: {size_bytes / (1024**3):.2f} GB" if size_bytes > 0 else f"     Size: {size_bytes} bytes")
    print(f"     Expected: {expected}")
    print(f"     Got:      {scores}")
    
    if not all_match:
        print(f"     MISMATCH!")
        for key in expected:
            if scores.get(key) != expected.get(key):
                print(f"       - {key}: expected {expected[key]}, got {scores.get(key)}")
    
    print()

print("=" * 100)
print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")

# Empty the log queue
while not log_queue.empty():
    log_queue.get()
