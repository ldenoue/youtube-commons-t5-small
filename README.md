# youtube-commons-t5-small

## install

python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# create dataset

`python youtube-commons.py`

# fine-tune model on the dataset

`python train_youtube_commons.py`

# try the fine-tuned model and compare its output to the non fine-tuned model

`python try_youtube_commons.py`

# convert the pytorch model to ONNX for Transformers.js

`python convert_youtube_common.py`

# test the ONNX model in the browser

`python3.13 -m http.server`
`open http://localhost:8000/youtube_common.html`


