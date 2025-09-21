import logging
import uuid

def configure_logging():
    """Configures structured logging for the batch processing system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler("batch_processing.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("BatchProcessor")

# Generate a unique run ID for each batch run
RUN_ID = str(uuid.uuid4())