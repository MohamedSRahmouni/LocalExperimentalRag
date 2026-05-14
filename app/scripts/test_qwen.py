# test_docling.py  (run: python test_docling.py)
"""Quick diagnostic for Docling availability"""

print("=" * 60)
print("🔍 DOCLING DIAGNOSTIC")
print("=" * 60)

# Test 1: Import
try:
    import docling
    print(f"✅ docling installed: {docling.__version__}")
except ImportError as e:
    print(f"❌ docling NOT installed: {e}")
    print("   Fix: pip install docling")

# Test 2: DocumentConverter
try:
    from docling.document_converter import DocumentConverter
    print("✅ DocumentConverter importable")
except ImportError as e:
    print(f"❌ DocumentConverter import failed: {e}")

# Test 3: Instantiate
try:
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()
    print("✅ DocumentConverter instantiated")
except Exception as e:
    print(f"❌ DocumentConverter init failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# Test 4: pandas + tabulate (needed for to_markdown)
try:
    import pandas as pd
    print(f"✅ pandas: {pd.__version__}")
    df = pd.DataFrame({"A": [1], "B": [2]})
    md = df.to_markdown(index=False)
    print(f"✅ to_markdown() works: {md[:30]}")
except ImportError as e:
    print(f"❌ pandas/tabulate issue: {e}")
    print("   Fix: pip install pandas tabulate")

# Test 5: pdfplumber fallback
try:
    import pdfplumber
    print(f"✅ pdfplumber available (fallback)")
except ImportError:
    print("❌ pdfplumber not available")

print("=" * 60)