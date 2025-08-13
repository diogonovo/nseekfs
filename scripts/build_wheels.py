#!/usr/bin/env python3
"""
Build Wheels Script for NSeekFS
Local testing of multi-platform wheel builds before CI/CD
"""

import os
import sys
import subprocess
import platform
import shutil
import tempfile
from pathlib import Path
import json
import time

def run_command(cmd, cwd=None, env=None, capture_output=True):
    """Run command and return result"""
    print(f"🔧 Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            env=env, 
            capture_output=capture_output,
            text=True,
            check=True
        )
        if capture_output:
            return result.stdout.strip()
        return ""
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if capture_output and e.stdout:
            print(f"STDOUT: {e.stdout}")
        if capture_output and e.stderr:
            print(f"STDERR: {e.stderr}")
        raise

def check_dependencies():
    """Check if required tools are installed"""
    print("🔍 Checking dependencies...")
    
    dependencies = {
        "python": "python --version",
        "rust": "rustc --version", 
        "cargo": "cargo --version",
        "maturin": "maturin --version"
    }
    
    missing = []
    for name, cmd in dependencies.items():
        try:
            version = run_command(cmd)
            print(f"  ✅ {name}: {version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ❌ {name}: NOT FOUND")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\n📦 Installation instructions:")
        if "maturin" in missing:
            print("  pip install maturin")
        if "rust" in missing or "cargo" in missing:
            print("  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        return False
    
    return True

def get_system_info():
    """Get current system information"""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(), 
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "architecture": platform.architecture()[0]
    }
    
    print("🖥️  System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return info

def build_source_distribution():
    """Build source distribution"""
    print("\n📦 Building source distribution...")
    
    # Clean previous builds
    if Path("dist").exists():
        shutil.rmtree("dist")
    
    try:
        output = run_command("maturin sdist --out dist")
        print(f"✅ Source distribution built")
        
        # List created files
        dist_files = list(Path("dist").glob("*.tar.gz"))
        for file in dist_files:
            size = file.stat().st_size / 1024 / 1024
            print(f"  📄 {file.name} ({size:.1f} MB)")
        
        return dist_files
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build source distribution: {e}")
        return []

def build_wheel_current_platform():
    """Build wheel for current platform"""
    print(f"\n🛞 Building wheel for current platform...")
    
    try:
        # Build wheel
        output = run_command("maturin build --release --out dist")
        print(f"✅ Wheel built for current platform")
        
        # List wheel files
        wheel_files = list(Path("dist").glob("*.whl"))
        for file in wheel_files:
            size = file.stat().st_size / 1024 / 1024
            print(f"  🛞 {file.name} ({size:.1f} MB)")
        
        return wheel_files
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build wheel: {e}")
        return []

def test_wheel_installation(wheel_path):
    """Test wheel installation in temporary environment"""
    print(f"\n🧪 Testing wheel installation: {wheel_path.name}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "test_env"
        
        try:
            # Create virtual environment
            run_command(f"python -m venv {venv_path}")
            
            # Determine activation script path
            if platform.system() == "Windows":
                activate_script = venv_path / "Scripts" / "activate.bat"
                pip_exe = venv_path / "Scripts" / "pip.exe"
                python_exe = venv_path / "Scripts" / "python.exe"
            else:
                activate_script = venv_path / "bin" / "activate" 
                pip_exe = venv_path / "bin" / "pip"
                python_exe = venv_path / "bin" / "python"
            
            # Install numpy first (required dependency)
            run_command([str(pip_exe), "install", "numpy"])
            print("  ✅ Numpy installed")
            
            # Install our wheel
            run_command([str(pip_exe), "install", str(wheel_path)])
            print(f"  ✅ {wheel_path.name} installed")
            
            # Test import and basic functionality
            test_script = """
import sys
print(f"Python: {sys.version}")

try:
    import nseekfs
    print(f"NSeekFS version: {nseekfs.__version__}")
    
    import numpy as np
    print("Testing basic functionality...")
    
    # Create test data
    embeddings = np.random.randn(100, 64).astype(np.float32)
    
    # Create index
    index = nseekfs.from_embeddings(embeddings)
    
    # Test query
    results = index.query(embeddings[0], top_k=5)
    
    assert len(results) == 5
    assert results[0]['idx'] == 0
    assert 0.99 <= results[0]['score'] <= 1.01
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
            
            # Write test script to temp file
            test_file = Path(temp_dir) / "test_nseekfs.py"
            test_file.write_text(test_script)
            
            # Run test
            run_command([str(python_exe), str(test_file)], capture_output=False)
            print("  ✅ Wheel functionality test passed!")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Wheel test failed: {e}")
            return False

def simulate_cibuildwheel():
    """Simulate cibuildwheel process locally"""
    print("\n🔄 Simulating cibuildwheel process...")
    
    # This is what cibuildwheel would do on different platforms
    system = platform.system().lower()
    
    if system == "linux":
        print("🐧 Linux platform detected")
        print("  Note: Full manylinux builds require Docker")
        print("  Current build will be linux_x86_64 (not manylinux)")
        
    elif system == "darwin":
        print("🍎 macOS platform detected")
        arch = platform.machine()
        if arch == "arm64":
            print("  Building for Apple Silicon (M1/M2)")
        else:
            print("  Building for Intel x86_64")
            
    elif system == "windows":
        print("🪟 Windows platform detected")
        print("  Building for win_amd64")
    
    else:
        print(f"⚠️  Unknown platform: {system}")
    
    # Build wheel with current settings
    wheel_files = build_wheel_current_platform()
    
    return wheel_files

def validate_wheel_metadata(wheel_path):
    """Validate wheel metadata"""
    print(f"\n🔍 Validating wheel metadata: {wheel_path.name}")
    
    try:
        # Use wheel command to show metadata
        try:
            metadata = run_command(f"python -m wheel unpack --dest temp_wheel {wheel_path}")
        except:
            # Alternative: extract and read METADATA manually
            import zipfile
            with zipfile.ZipFile(wheel_path, 'r') as zip_file:
                # Find METADATA file
                metadata_files = [f for f in zip_file.namelist() if f.endswith('METADATA')]
                if metadata_files:
                    metadata_content = zip_file.read(metadata_files[0]).decode('utf-8')
                    print("  📄 Wheel metadata:")
                    
                    # Parse key metadata fields
                    for line in metadata_content.split('\n')[:20]:  # First 20 lines
                        if line.startswith(('Name:', 'Version:', 'Summary:', 'Author:', 'Platform:')):
                            print(f"    {line}")
                    
                    print("  ✅ Metadata looks good")
                    return True
                else:
                    print("  ❌ No METADATA file found")
                    return False
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Could not validate metadata: {e}")
        return False

def check_wheel_compatibility():
    """Check wheel naming and compatibility"""
    print("\n🎯 Checking wheel compatibility...")
    
    wheel_files = list(Path("dist").glob("*.whl"))
    
    for wheel_file in wheel_files:
        print(f"\n🛞 Analyzing: {wheel_file.name}")
        
        # Parse wheel filename
        parts = wheel_file.stem.split('-')
        if len(parts) >= 5:
            name = parts[0]
            version = parts[1] 
            python_tag = parts[2]
            abi_tag = parts[3]
            platform_tag = parts[4]
            
            print(f"  📦 Package: {name}")
            print(f"  🔢 Version: {version}")
            print(f"  🐍 Python: {python_tag}")
            print(f"  🔗 ABI: {abi_tag}")
            print(f"  🖥️  Platform: {platform_tag}")
            
            # Check if this looks correct
            expected_patterns = {
                "linux": ["linux_x86_64", "manylinux"],
                "darwin": ["macosx_", "universal2"],
                "windows": ["win_amd64", "win32"]
            }
            
            current_system = platform.system().lower()
            if current_system in expected_patterns:
                expected = expected_patterns[current_system]
                if any(pattern in platform_tag for pattern in expected):
                    print(f"  ✅ Platform tag looks correct for {current_system}")
                else:
                    print(f"  ⚠️  Platform tag might be unexpected: {platform_tag}")
            
        else:
            print(f"  ❌ Unexpected wheel filename format")

def generate_build_report():
    """Generate build report"""
    print("\n📊 Generating build report...")
    
    dist_path = Path("dist")
    if not dist_path.exists():
        print("❌ No dist directory found")
        return
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": get_system_info(),
        "files": [],
        "total_size_mb": 0
    }
    
    # Analyze all distribution files
    for file_path in dist_path.iterdir():
        if file_path.is_file():
            size_mb = file_path.stat().st_size / 1024 / 1024
            report["files"].append({
                "name": file_path.name,
                "size_mb": round(size_mb, 2),
                "type": "wheel" if file_path.suffix == ".whl" else "source"
            })
            report["total_size_mb"] += size_mb
    
    report["total_size_mb"] = round(report["total_size_mb"], 2)
    
    # Save report
    report_file = Path("build_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Report saved to: {report_file}")
    
    # Print summary
    print(f"\n📋 Build Summary:")
    print(f"  Total files: {len(report['files'])}")
    print(f"  Total size: {report['total_size_mb']} MB")
    print(f"  Wheels: {sum(1 for f in report['files'] if f['type'] == 'wheel')}")
    print(f"  Source: {sum(1 for f in report['files'] if f['type'] == 'source')}")

def main():
    """Main build process"""
    print("🚀 NSeekFS Wheel Building Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("Cargo.toml").exists():
        print("❌ Cargo.toml not found. Run this script from the project root.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get system info
    system_info = get_system_info()
    
    try:
        # Build source distribution
        print("\n" + "=" * 50)
        sdist_files = build_source_distribution()
        
        # Build wheel for current platform
        print("\n" + "=" * 50)
        wheel_files = simulate_cibuildwheel()
        
        if wheel_files:
            # Validate wheels
            print("\n" + "=" * 50)
            check_wheel_compatibility()
            
            # Test first wheel
            if wheel_files:
                print("\n" + "=" * 50)
                validate_wheel_metadata(wheel_files[0])
                
                print("\n" + "=" * 50)
                test_success = test_wheel_installation(wheel_files[0])
                
                if not test_success:
                    print("❌ Wheel testing failed")
                    sys.exit(1)
        
        # Generate report
        print("\n" + "=" * 50)
        generate_build_report()
        
        print("\n🎉 Build process completed successfully!")
        print("\n📁 Distribution files created in ./dist/")
        print("🧪 Ready for CI/CD pipeline testing!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()