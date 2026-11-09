"""FastAPI application entrypoint utilizing application factory pattern."""

import logging
import uvicorn
from src.api.app import create_app
from src.api.service import model_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Primary app instance constructed via factory method
app = create_app(manager=model_manager)


def run():
    """Runs uvicorn development server."""
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
