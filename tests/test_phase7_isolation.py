import ast
import os
import pytest

def get_all_python_files(directory, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = ['tests', 'configs', 'results', '.venv', '.git', '__pycache__', 'envs', 'simulator']
    
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Exclude directories we don't want to scan (like tests where imports are allowed, or venv)
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_imports_in_file(filepath, forbidden_modules):
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=filepath)
        
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for f_mod in forbidden_modules:
                    if alias.name == f_mod or alias.name.startswith(f_mod + '.'):
                        violations.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for f_mod in forbidden_modules:
                    if node.module == f_mod or node.module.startswith(f_mod + '.'):
                        violations.append((node.lineno, node.module))
    return violations

def test_no_data_leakage():
    # Only RL training/tuning code must be verified.
    # So we scan rl/ directory.
    rl_files = get_all_python_files('rl', exclude_dirs=[])
    
    # We must also scan root-level training scripts like run_phase5.py
    root_files = [f for f in os.listdir('.') if f.endswith('.py') and 'run' in f and 'phase7' not in f]
    
    all_files = rl_files + root_files
    
    forbidden = ['configs.eval_scenarios', 'envs.scenario_generators']
    
    all_violations = []
    for filepath in all_files:
        if not os.path.isfile(filepath):
            continue
        v = check_imports_in_file(filepath, forbidden)
        if v:
            all_violations.append((filepath, v))
            
    assert len(all_violations) == 0, f"Data leakage detected! Forbidden imports found: {all_violations}"
