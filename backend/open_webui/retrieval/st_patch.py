import logging
import sys
import os
import importlib.util

log = logging.getLogger(__name__)

def patch_sentence_transformers():
    """
    Attempt to patch issues with sentence_transformers imports
    by addressing common dependencies before importing.
    """
    # Check if we need the patch
    if importlib.util.find_spec("sentence_transformers") is None:
        log.warning("sentence_transformers package not found")
        return False
        
    # Try to patch _lzma first if it's missing
    if '_lzma' not in sys.modules:
        try:
            # Try to use backports.lzma
            import backports.lzma as _lzma
            sys.modules['_lzma'] = _lzma
            log.info("Successfully loaded backports.lzma as _lzma")
        except ImportError:
            log.warning("Could not import backports.lzma")
            try:
                # Try to install it on the fly
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "backports.lzma"])
                import backports.lzma as _lzma
                sys.modules['_lzma'] = _lzma
                log.info("Successfully installed and loaded backports.lzma")
            except Exception as e:
                log.error(f"Failed to install backports.lzma: {e}")
                # Create a minimal mock as last resort
                class MockLZMA:
                    def __getattr__(self, name):
                        return lambda *args, **kwargs: None
                
                sys.modules['_lzma'] = MockLZMA()
                log.info("Created mock _lzma module")
    
    # Now try to safely import sentence_transformers
    try:
        import sentence_transformers
        log.info(f"Successfully imported sentence_transformers {sentence_transformers.__version__}")
        return True
    except ImportError as e:
        log.error(f"Failed to import sentence_transformers: {e}")
        return False
        
def safe_import_sentence_transformers():
    """
    Safely import SentenceTransformer with patching.
    Returns the SentenceTransformer class or None if not available.
    """
    if not patch_sentence_transformers():
        return None
        
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError as e:
        log.error(f"Could not import SentenceTransformer: {e}")
        return None
