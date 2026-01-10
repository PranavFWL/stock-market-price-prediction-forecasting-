import sys

print("Checking data_fetcher.py...")
try:
    with open('data_fetcher.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'data_fetcher.py', 'exec')
    print("✓ data_fetcher.py compiles successfully")
    
    # Try to execute it
    exec(code)
    print("✓ data_fetcher.py executes successfully")
    
except SyntaxError as e:
    print(f"✗ SYNTAX ERROR in data_fetcher.py:")
    print(f"  Line {e.lineno}: {e.msg}")
    print(f"  {e.text}")
except Exception as e:
    print(f"✗ ERROR in data_fetcher.py: {e}")

print("\nChecking model_architecture.py...")
try:
    with open('model_architecture.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'model_architecture.py', 'exec')
    print("✓ model_architecture.py compiles successfully")
    
except SyntaxError as e:
    print(f"✗ SYNTAX ERROR in model_architecture.py:")
    print(f"  Line {e.lineno}: {e.msg}")
    print(f"  {e.text}")
except Exception as e:
    print(f"✗ ERROR in model_architecture.py: {e}")

print("\nChecking config.py...")
try:
    from config import DATA_CONFIG, MODEL_CONFIG
    print("✓ config.py imports successfully")
    print(f"  DATA_CONFIG keys: {list(DATA_CONFIG.keys())}")
except Exception as e:
    print(f"✗ ERROR in config.py: {e}")