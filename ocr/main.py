import paddle
from flask import Flask
from flask_compress import Compress
from flask_cors import CORS
from loguru import logger

from api.job import ocr  # fmt: skip # noqa
from tasks.pdf2txt import init

paddle.disable_signal_handler()

logger.add("logs/app.log", rotation="01:00", retention="1 months")
init()

# os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

app = Flask(__name__)
CORS(app)
Compress(app)

app.register_blueprint(ocr)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
