import logging
from fastapi import Request, HTTPException, status

from open_webui.routers.retrieval import get_ef, get_embedding_function
from open_webui.config import RAG_EMBEDDING_MODEL_AUTO_UPDATE

log = logging.getLogger(__name__)

def ensure_embedding_function(request: Request):
    """
    Ensure that the embedding function is initialized.
    This function can be called before any operation that requires embeddings.
    """
    if request.app.state.EMBEDDING_FUNCTION is None or request.app.state.ef is None:
        log.warning("Embedding function not initialized. Attempting to initialize.")
        try:
            # Try patching sentence_transformers first if we need it
            if request.app.state.config.RAG_EMBEDDING_ENGINE == "":
                from open_webui.retrieval.st_patch import patch_sentence_transformers
                patch_sentence_transformers()
                
            # Initialize ef if needed
            if request.app.state.ef is None:
                log.info(f"Initializing ef with engine: {request.app.state.config.RAG_EMBEDDING_ENGINE}, model: {request.app.state.config.RAG_EMBEDDING_MODEL}")
                request.app.state.ef = get_ef(
                    request.app.state.config.RAG_EMBEDDING_ENGINE,
                    request.app.state.config.RAG_EMBEDDING_MODEL,
                    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
                )
                
                # If ef is still None after get_ef, try the fallback
                if request.app.state.ef is None:
                    log.warning("Primary embedding function initialization failed. Trying fallback.")
                    try:
                        from open_webui.retrieval.fallback_embeddings import get_fallback_embedding_model
                        request.app.state.ef = get_fallback_embedding_model()
                        log.info("Using fallback embedding model")
                    except Exception as fallback_error:
                        log.error(f"Failed to initialize fallback embedding model: {fallback_error}")
            
            # Initialize EMBEDDING_FUNCTION if ef is available
            if request.app.state.ef is not None:
                try:
                    request.app.state.EMBEDDING_FUNCTION = get_embedding_function(
                        request.app.state.config.RAG_EMBEDDING_ENGINE,
                        request.app.state.config.RAG_EMBEDDING_MODEL,
                        request.app.state.ef,
                        (
                            request.app.state.config.RAG_OPENAI_API_BASE_URL
                            if request.app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                            else request.app.state.config.RAG_OLLAMA_BASE_URL
                        ),
                        (
                            request.app.state.config.RAG_OPENAI_API_KEY
                            if request.app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                            else request.app.state.config.RAG_OLLAMA_API_KEY
                        ),
                        request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
                    )
                except Exception as e:
                    log.error(f"Error in get_embedding_function: {e}")
                    # Direct fallback if get_embedding_function fails
                    if hasattr(request.app.state.ef, 'encode'):
                        log.info("Creating direct wrapper for embedding function")
                        request.app.state.EMBEDDING_FUNCTION = lambda query, prefix=None, user=None: request.app.state.ef.encode(
                            query, **({"prompt": prefix} if prefix else {})
                        )
            
            # Final fallback if everything else fails
            if request.app.state.EMBEDDING_FUNCTION is None:
                log.warning("Creating final fallback embedding function")
                from open_webui.retrieval.fallback_embeddings import get_fallback_embedding_model
                fallback_model = get_fallback_embedding_model()
                request.app.state.EMBEDDING_FUNCTION = lambda query, prefix=None, user=None: fallback_model.encode(
                    query, **({"prompt": prefix} if prefix else {})
                )
                
            log.info("Successfully initialized embedding function")
            return True
        except Exception as e:
            log.error(f"Error initializing embedding function: {e}")
            # Don't raise an exception - provide a minimal working function
            try:
                from open_webui.retrieval.fallback_embeddings import get_fallback_embedding_model
                fallback_model = get_fallback_embedding_model()
                request.app.state.EMBEDDING_FUNCTION = lambda query, prefix=None, user=None: fallback_model.encode(
                    query, **({"prompt": prefix} if prefix else {})
                )
                log.info("Created minimal fallback embedding function after error")
                return True
            except Exception as final_error:
                log.error(f"Fatal error initializing any embedding function: {final_error}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize embedding function: {str(e)}"
                )
    
    return True
