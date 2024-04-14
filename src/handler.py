import base64
import mimetypes
import time
import os
import json
import subprocess
import uuid
from datetime import datetime

import runpod
import requests
from firebase_admin import credentials, initialize_app, storage, firestore

from requests.adapters import HTTPAdapter, Retry
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger

logger = RunPodLogger()

SERVICE_CERT = json.loads(os.environ["FIREBASE_KEY"])
SADTALKER_SERVICE_CERT = json.loads(os.environ["SADTALKER_FIREBASE_KEY"])
STORAGE_BUCKET = os.environ["STORAGE_BUCKET"]

cred_obj = credentials.Certificate(SERVICE_CERT)
sad_cred_obj = credentials.Certificate(SADTALKER_SERVICE_CERT)

default_app = initialize_app(cred_obj, {"storageBucket": STORAGE_BUCKET}, name="ImagineCrafter")
sad_app = initialize_app(sad_cred_obj, name='sadtalker')

LOCAL_URL = "http://127.0.0.1:5000"

cog_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
cog_session.mount('http://', HTTPAdapter(max_retries=retries))

INPUT_SCHEMA = {
    "top_k": {
        "type": int,
        # "title": "Top K",
        "default": 250,
        "required": False,
        # "x-order": 2,
        # "description": "Reduces sampling to the k most likely tokens."
    },
    "top_p": {
        "type": float,
        # "title": "Top P",
        "default": 0,
        "required": False,
        # "x-order": 3,
        # "description": "Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used."
    },
    "prompt": {
        "type": str,
        "title": "Prompt",
        "required": True,
        # "x-order": 0,
        "description": "Prompt that describes the sound"
    },
    "duration": {
        "type": float,
        # "title": "Duration",
        "default": 3,
        "required": False,
        # "maximum": 10,
        # "minimum": 1,
        # "x-order": 1,
        # "description": "Max duration of the sound",
        "constraints": lambda duration: 1 <= duration <= 10,
    },
    "temperature": {
        "type": float,
        # "title": "Temperature",
        "default": 1,
        "required": False,
        # "x-order": 4,
        # "description": "Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity."
    },
    "output_format": {
        "constraint": lambda output_format: output_format in [
            "wav",
            "mp3"
        ],
        "type": str,
        # "title": "output_format",
        # "description": "Output format for generated audio.",
        "default": "mp3",
        "required": False,
        # "x-order": 6
    },
    "classifier_free_guidance": {
        "type": int,
        # "title": "Classifier Free Guidance",
        "default": 3,
        "required": False,
        # "x-order": 5,
        # "description": "Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs."
    },
    "user_id": {
        "type": str,
        "required": True
    },
}

# ----------------------------- Start API Service ---------------------------- #
# Call "python -m cog.server.http" in a subprocess to start the API service.
subprocess.Popen(["python", "-m", "cog.server.http"])


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            health = requests.get(url, timeout=120)
            status = health.json()["status"]

            if status == "READY":
                time.sleep(1)
                return

        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def run_inference(inference_request):
    '''
    Run inference on a request.
    '''
    response = cog_session.post(url=f'{LOCAL_URL}/predictions',
                                json=inference_request, timeout=600)
    return response.json()


def get_extension_from_mime(mime_type):
    extension = mimetypes.guess_extension(mime_type)
    return extension


def to_firestore(file_url, user_id):
    db = firestore.client(app=sad_app)

    current_utc_time = datetime.utcnow()
    formatted_time = current_utc_time.strftime("%Y-%m-%d %H:%M:%S")

    push_data = {
        "uploaderId": user_id,
        # "videoCaption": prompt,
        "audioUrl": file_url,
        "timestamp": formatted_time
    }

    collection_path = "audioList"

    print("*************Starting firestore data push***************")
    update_time, firestore_push_id = db.collection(collection_path).add(
        push_data
    )

    print(update_time, firestore_push_id)


def upload_file(filename, folder='audiogen'):
    destination_blob_name = f'{folder}/{filename}'
    bucket = storage.bucket(app=default_app)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(filename)

    # Opt : if you want to make public access from the URL
    blob.make_public()

    logger.info(f"{filename}: Uploaded to firebase...")
    return blob.public_url


def to_file(data: str):
    # bs4_code = data.split(';base64,')[-1]

    # Splitting the input string to get the MIME type and the base64 data
    split_data = data.split(",")
    mime_type = split_data[0].split(":")[1].split(';')[0]
    base64_data = split_data[1]

    ext = get_extension_from_mime(mime_type)
    f_name = f'{uuid.uuid4()}'
    filename = f'{f_name}{ext}'
    decoded_data = base64.b64decode(base64_data)

    with open(filename, 'wb') as f:
        f.write(decoded_data)

    file_url = upload_file(filename)

    return file_url



# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''

    validated_input = validate(event['input'], INPUT_SCHEMA)

    if 'errors' in validated_input:
        logger.error('Error in input...')
        return {
            'errors': validated_input['errors']
        }

    logger.info(f'Received event: {validated_input}')

    result = run_inference({"input": event["input"]})

    file_url = to_file(data=result['output'])

    to_firestore(file_url, event['input']['user_id'])

    return {
        'user_id': event['input']['user_id'],
        'audio_url': file_url,
        'prompt': event['input']['prompt']
    }


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/health-check')

    print("Cog API Service is ready. Starting RunPod serverless handler...")

    runpod.serverless.start({"handler": handler})
