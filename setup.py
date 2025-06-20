#!/usr/bin/env python3
"""
Setup Script for Text Intensification Tool
Automatically installs all required dependencies including spaCy models.
"""

import subprocess
import sys
import importlib.util


def check_package_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def install_package(package_name):
    """Install a package using pip."""
    print(f"📦 Installing {package_name}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name
        ], stdout=subprocess.DEVNULL)
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False


def download_spacy_model(model_name):
    """Download a spaCy model."""
    print(f"📥 Downloading spaCy model: {model_name}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", model_name
        ], stdout=subprocess.DEVNULL)
        print(f"✅ Successfully downloaded {model_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to download {model_name}")
        return False


def check_spacy_model(model_name):
    """Check if a spaCy model is available."""
    try:
        import spacy
        spacy.load(model_name)
        return True
    except (ImportError, OSError):
        return False


def main():
    """Main setup function."""
    print("🔧 TEXT INTENSIFICATION TOOL - SETUP")
    print("=" * 50)
    print("This script will install all required dependencies.\n")

    # Required packages
    required_packages = [
        ("spacy", "spacy>=3.4.0"),
        ("numpy", "numpy>=1.21.0"),
    ]

    # Install required packages
    print("📋 Checking Python packages...")
    all_packages_ok = True

    for package_name, pip_name in required_packages:
        if check_package_installed(package_name):
            print(f"✅ {package_name} is already installed")
        else:
            print(f"⚠️  {package_name} not found")
            if install_package(pip_name):
                print(f"✅ {package_name} installed successfully")
            else:
                all_packages_ok = False

    if not all_packages_ok:
        print("\n❌ Some packages failed to install. Please install manually:")
        print("pip install spacy numpy")
        return False

    # Check and download spaCy models
    print(f"\n🤖 Checking spaCy language models...")

    # Try to download the large model first
    if check_spacy_model("en_core_web_lg"):
        print("✅ en_core_web_lg is already available")
    else:
        print("⚠️  en_core_web_lg not found, attempting download...")
        if download_spacy_model("en_core_web_lg"):
            print("✅ en_core_web_lg downloaded successfully")
        else:
            print("⚠️  Could not download en_core_web_lg, trying smaller model...")

            # Fall back to small model
            if check_spacy_model("en_core_web_sm"):
                print("✅ en_core_web_sm is already available")
            else:
                print("📥 Downloading en_core_web_sm...")
                if download_spacy_model("en_core_web_sm"):
                    print("✅ en_core_web_sm downloaded successfully")
                else:
                    print("❌ Failed to download any spaCy model")
                    print("Please run manually: python -m spacy download en_core_web_sm")
                    return False

    # Test the installation
    print(f"\n🧪 Testing installation...")
    try:
        # Import and test the main module
        if check_package_installed("compare_texts") or check_package_installed("main"):
            try:
                # Try importing from main.py
                from main import TextIntensificationComparator
                print("✅ Found main.py")
                module_name = "main.py"
            except ImportError:
                try:
                    # Try importing from compare_texts.py
                    from compare_texts import TextIntensificationComparator
                    print("✅ Found compare_texts.py")
                    module_name = "compare_texts.py"
                except ImportError:
                    print("⚠️  Could not find main module (main.py or compare_texts.py)")
                    print("Make sure the main script is in the current directory")
                    return False

            # Test creating the comparator
            print("🔧 Testing comparator initialization...")
            comparator = TextIntensificationComparator()
            print("✅ Comparator initialized successfully")

            # Test with a simple analysis
            test_text = "This is a significant and remarkable test."
            result = comparator.analyze_text(test_text, "Test")
            print(f"✅ Analysis test completed - found {result['intensifying_adjectives']} intensifying adjectives")

        else:
            print("⚠️  Main module not found in current directory")
            print("Make sure main.py or compare_texts.py is present")
            return False

    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

    # Success message
    print(f"\n🎉 SETUP COMPLETE!")
    print("=" * 30)
    print("✅ All dependencies installed")
    print("✅ spaCy model available")
    print(f"✅ Main module ({module_name}) working")
    print()
    print("You can now run:")
    print(f"  python {module_name}           # Interactive mode")
    print("  python tests.py              # Run test suite")
    print()
    print("For better accuracy, consider upgrading to the large model:")
    print("  python -m spacy download en_core_web_lg")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print(f"\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print(f"\n🚀 Ready to use! Run 'python main.py' or 'python compare_texts.py'")
        sys.exit(0)