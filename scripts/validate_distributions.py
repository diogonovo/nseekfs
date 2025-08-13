#!/usr/bin/env python3
"""
Distribution Validation Script
Validates wheels and source distributions before upload
"""

import os
import sys
import subprocess
import zipfile
import tarfile
import tempfile
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

def run_command(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run command and return result"""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

class DistributionValidator:
    """Validates Python distribution packages"""
    
    def __init__(self, dist_dir: Path):
        self.dist_dir = Path(dist_dir)
        self.validation_results = {
            "wheels": [],
            "source_distributions": [],
            "summary": {"total": 0, "passed": 0, "failed": 0},
            "errors": []
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Validate all distributions in the directory"""
        print("🔍 Validating distributions...")
        
        if not self.dist_dir.exists():
            raise FileNotFoundError(f"Distribution directory not found: {self.dist_dir}")
        
        # Find all distribution files
        wheels = list(self.dist_dir.glob("*.whl"))
        sdists = list(self.dist_dir.glob("*.tar.gz"))
        
        print(f"Found {len(wheels)} wheels and {len(sdists)} source distributions")
        
        # Validate wheels
        for wheel_path in wheels:
            try:
                result = self.validate_wheel(wheel_path)
                self.validation_results["wheels"].append(result)
                if result["valid"]:
                    self.validation_results["summary"]["passed"] += 1
                else:
                    self.validation_results["summary"]["failed"] += 1
            except Exception as e:
                error = f"Wheel validation failed for {wheel_path.name}: {e}"
                self.validation_results["errors"].append(error)
                self.validation_results["summary"]["failed"] += 1
                print(f"❌ {error}")
        
        # Validate source distributions
        for sdist_path in sdists:
            try:
                result = self.validate_sdist(sdist_path)
                self.validation_results["source_distributions"].append(result)
                if result["valid"]:
                    self.validation_results["summary"]["passed"] += 1
                else:
                    self.validation_results["summary"]["failed"] += 1
            except Exception as e:
                error = f"Sdist validation failed for {sdist_path.name}: {e}"
                self.validation_results["errors"].append(error)
                self.validation_results["summary"]["failed"] += 1
                print(f"❌ {error}")
        
        self.validation_results["summary"]["total"] = len(wheels) + len(sdists)
        
        return self.validation_results
    
    def validate_wheel(self, wheel_path: Path) -> Dict[str, Any]:
        """Validate a wheel file"""
        print(f"🛞 Validating wheel: {wheel_path.name}")
        
        result = {
            "filename": wheel_path.name,
            "type": "wheel",
            "valid": True,
            "size_mb": round(wheel_path.stat().st_size / 1024 / 1024, 2),
            "checks": {},
            "metadata": {},
            "issues": []
        }
        
        try:
            # Check 1: Wheel file structure
            result["checks"]["structure"] = self._check_wheel_structure(wheel_path)
            
            # Check 2: Metadata validation
            result["checks"]["metadata"] = self._check_wheel_metadata(wheel_path, result)
            
            # Check 3: Platform tag validation
            result["checks"]["platform_tag"] = self._check_platform_tag(wheel_path)
            
            # Check 4: Python version compatibility
            result["checks"]["python_version"] = self._check_python_version(wheel_path)
            
            # Check 5: File integrity
            result["checks"]["integrity"] = self._check_file_integrity(wheel_path)
            
            # Overall validation
            all_checks_passed = all(result["checks"].values())
            result["valid"] = all_checks_passed
            
            if all_checks_passed:
                print(f"  ✅ {wheel_path.name} validation passed")
            else:
                print(f"  ❌ {wheel_path.name} validation failed")
                for check, passed in result["checks"].items():
                    if not passed:
                        print(f"    ❌ {check} check failed")
        
        except Exception as e:
            result["valid"] = False
            result["issues"].append(f"Validation error: {e}")
            print(f"  ❌ Validation error: {e}")
        
        return result
    
    def validate_sdist(self, sdist_path: Path) -> Dict[str, Any]:
        """Validate a source distribution"""
        print(f"📦 Validating source distribution: {sdist_path.name}")
        
        result = {
            "filename": sdist_path.name,
            "type": "source",
            "valid": True,
            "size_mb": round(sdist_path.stat().st_size / 1024 / 1024, 2),
            "checks": {},
            "contents": [],
            "issues": []
        }
        
        try:
            # Check 1: Tarball structure
            result["checks"]["structure"] = self._check_sdist_structure(sdist_path, result)
            
            # Check 2: Required files present
            result["checks"]["required_files"] = self._check_required_files(sdist_path)
            
            # Check 3: No unwanted files
            result["checks"]["clean_contents"] = self._check_clean_contents(sdist_path)
            
            # Check 4: File integrity
            result["checks"]["integrity"] = self._check_file_integrity(sdist_path)
            
            # Overall validation
            all_checks_passed = all(result["checks"].values())
            result["valid"] = all_checks_passed
            
            if all_checks_passed:
                print(f"  ✅ {sdist_path.name} validation passed")
            else:
                print(f"  ❌ {sdist_path.name} validation failed")
        
        except Exception as e:
            result["valid"] = False
            result["issues"].append(f"Validation error: {e}")
            print(f"  ❌ Validation error: {e}")
        
        return result
    
    def _check_wheel_structure(self, wheel_path: Path) -> bool:
        """Check wheel internal structure"""
        try:
            with zipfile.ZipFile(wheel_path, 'r') as zf:
                files = zf.namelist()
                
                # Check for required files
                has_metadata = any(f.endswith('.dist-info/METADATA') for f in files)
                has_wheel = any(f.endswith('.dist-info/WHEEL') for f in files)
                has_record = any(f.endswith('.dist-info/RECORD') for f in files)
                
                return has_metadata and has_wheel and has_record
                
        except Exception as e:
            print(f"    ❌ Structure check failed: {e}")
            return False
    
    def _check_wheel_metadata(self, wheel_path: Path, result: Dict) -> bool:
        """Check wheel metadata"""
        try:
            with zipfile.ZipFile(wheel_path, 'r') as zf:
                # Find METADATA file
                metadata_files = [f for f in zf.namelist() if f.endswith('METADATA')]
                if not metadata_files:
                    return False
                
                metadata_content = zf.read(metadata_files[0]).decode('utf-8')
                
                # Parse key metadata
                metadata = {}
                for line in metadata_content.split('\n'):
                    if ':' in line and not line.startswith(' '):
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                
                result["metadata"] = metadata
                
                # Check required fields
                required_fields = ['Name', 'Version', 'Author']
                return all(field in metadata for field in required_fields)
                
        except Exception as e:
            print(f"    ❌ Metadata check failed: {e}")
            return False
    
    def _check_platform_tag(self, wheel_path: Path) -> bool:
        """Check platform tag validity"""
        filename = wheel_path.name
        
        # Parse wheel filename: name-version-python-abi-platform.whl
        parts = filename.replace('.whl', '').split('-')
        if len(parts) < 5:
            return False
        
        platform_tag = parts[-1]
        
        # Valid platform tags
        valid_platforms = [
            'linux_x86_64', 'linux_aarch64',
            'manylinux1_x86_64', 'manylinux2010_x86_64', 'manylinux2014_x86_64',
            'macosx_10_9_x86_64', 'macosx_11_0_arm64', 'macosx_10_9_universal2',
            'win32', 'win_amd64',
            'any'
        ]
        
        # Check if platform tag is valid or follows expected pattern
        return (platform_tag in valid_platforms or 
                platform_tag.startswith(('manylinux', 'macosx_', 'linux_', 'win')))
    
    def _check_python_version(self, wheel_path: Path) -> bool:
        """Check Python version compatibility"""
        filename = wheel_path.name
        parts = filename.replace('.whl', '').split('-')
        
        if len(parts) < 5:
            return False
        
        python_tag = parts[2]
        
        # Valid Python tags
        valid_python_tags = [
            'py3', 'py38', 'py39', 'py310', 'py311', 'py312',
            'cp38', 'cp39', 'cp310', 'cp311', 'cp312'
        ]
        
        return python_tag in valid_python_tags or python_tag.startswith(('py', 'cp'))
    
    def _check_sdist_structure(self, sdist_path: Path, result: Dict) -> bool:
        """Check source distribution structure"""
        try:
            with tarfile.open(sdist_path, 'r:gz') as tf:
                files = tf.getnames()
                result["contents"] = files[:20]  # First 20 files for inspection
                
                # Check for required files
                has_setup_or_pyproject = any(
                    f.endswith(('setup.py', 'pyproject.toml', 'setup.cfg')) 
                    for f in files
                )
                
                return has_setup_or_pyproject
                
        except Exception as e:
            print(f"    ❌ Sdist structure check failed: {e}")
            return False
    
    def _check_required_files(self, sdist_path: Path) -> bool:
        """Check for required files in source distribution"""
        try:
            with tarfile.open(sdist_path, 'r:gz') as tf:
                files = tf.getnames()
                
                # Required files
                has_readme = any('README' in f.upper() for f in files)
                has_license = any('LICENSE' in f.upper() for f in files)
                has_pyproject = any(f.endswith('pyproject.toml') for f in files)
                
                return has_readme and has_license and has_pyproject
                
        except Exception:
            return False
    
    def _check_clean_contents(self, sdist_path: Path) -> bool:
        """Check that source distribution doesn't contain unwanted files"""
        try:
            with tarfile.open(sdist_path, 'r:gz') as tf:
                files = tf.getnames()
                
                # Files that shouldn't be in source distribution
                unwanted_patterns = [
                    '__pycache__', '.pyc', '.pyo', '.DS_Store',
                    'target/', '.git/', 'Cargo.lock', '.env'
                ]
                
                for file in files:
                    for pattern in unwanted_patterns:
                        if pattern in file:
                            print(f"    ⚠️  Unwanted file in sdist: {file}")
                            return False
                
                return True
                
        except Exception:
            return False
    
    def _check_file_integrity(self, file_path: Path) -> bool:
        """Check file integrity"""
        try:
            # Calculate SHA256 hash
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            # File should not be empty and should have reasonable size
            size = file_path.stat().st_size
            return size > 1024  # At least 1KB
            
        except Exception:
            return False
    
    def generate_report(self, output_file: Optional[Path] = None) -> None:
        """Generate validation report"""
        if output_file is None:
            output_file = Path("distribution_validation_report.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"📄 Validation report saved: {output_file}")
    
    def print_summary(self) -> None:
        """Print validation summary"""
        summary = self.validation_results["summary"]
        
        print("\n📊 Validation Summary")
        print("=" * 30)
        print(f"Total distributions: {summary['total']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        
        if summary["failed"] > 0:
            print(f"\n❌ Validation failed for {summary['failed']} distributions")
            for error in self.validation_results["errors"]:
                print(f"  • {error}")
        else:
            print(f"\n🎉 All {summary['total']} distributions passed validation!")

def main():
    """Main validation process"""
    print("🔍 NSeekFS Distribution Validator")
    print("=" * 40)
    
    # Find distribution directory
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ dist/ directory not found")
        print("Run 'make wheels' or 'python scripts/build_wheels.py' first")
        sys.exit(1)
    
    # Run validation
    validator = DistributionValidator(dist_dir)
    results = validator.validate_all()
    
    # Generate report
    validator.generate_report()
    validator.print_summary()
    
    # Exit with appropriate code
    if results["summary"]["failed"] > 0:
        print("\n❌ Validation failed! Fix issues before uploading.")
        sys.exit(1)
    else:
        print("\n✅ All distributions are valid and ready for upload!")
        sys.exit(0)

if __name__ == "__main__":
    main()