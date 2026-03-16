#!/usr/bin/env python
"""
Setup script for quick environment validation
"""

import os
import sys
import subprocess

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("⚠ Warning: Python 3.9+ is recommended")


def check_dependencies():
    """Check if key dependencies are installed"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'tensorflow',
        'flask', 'kafka', 'psycopg2', 'pytest'
    ]
    
    print("\nChecking dependencies...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")


def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'models',
        'logs',
        'notebooks'
    ]
    
    print("\nCreating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}")


def create_env_file():
    """Create .env file from template"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as f:
                content = f.read()
            with open('.env', 'w') as f:
                f.write(content)
            print("\n✓ .env file created from template")
        else:
            print("\n⚠ .env.example not found")


def main():
    """Run setup checks"""
    print("=" * 60)
    print("AI-Driven Healthcare Anomaly Detection System - Setup")
    print("=" * 60)
    
    check_python_version()
    check_dependencies()
    create_directories()
    create_env_file()
    
    print("\n" + "=" * 60)
    print("Setup complete! Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Add your healthcare dataset to data/ directory")
    print("3. Run: python scripts/validate_dataset.py")
    print("4. Run: python main.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
