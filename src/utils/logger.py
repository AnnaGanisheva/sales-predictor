import sys
import logging


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(name)s: %(module)s: %(message)s]",
    stream=sys.stdout
)
logger = logging.getLogger("Sales_Predictor")
