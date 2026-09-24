import os
import sys
import pytest
from streamlit.testing.v1 import AppTest

# Since Streamlit testing might require specific versions, and just in case it fails,
# we also provide a simple import test as a fallback smoke test.

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_app_imports_and_runs():
    """
    Very lightweight smoke test ensuring the app module can be parsed and executed 
    without SyntaxErrors or immediately failing top-level logic.
    """
    # Simply importing the file will execute top-level definitions and st.set_page_config
    # We must catch the Streamlit exception if it complains about running outside a context,
    # but the import itself proves there are no syntax errors or missing dependencies.
    try:
        import dashboard.app
    except Exception as e:
        # If it fails for reasons OTHER than Streamlit context issues, it's a real bug
        if "Streamlit" not in str(type(e)):
            raise e

def test_app_via_streamlit_testing():
    """Use Streamlit's official AppTest framework to mock a browser session."""
    app_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")
    
    try:
        at = AppTest.from_file(app_path)
        at.run(timeout=10)
        assert not at.exception, f"App crashed with: {at.exception}"
    except AttributeError:
        # In case older streamlit versions don't support AppTest API fully
        pytest.skip("Streamlit testing framework not fully supported in this version.")
