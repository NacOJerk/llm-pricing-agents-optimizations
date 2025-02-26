# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_predict_custom_trained_model_sample]
from typing import Dict, List, Union
import json

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    return response

first_instance = [
    
]

# [END aiplatform_predict_custom_trained_model_sample]
with open('.secrets/endpoint_details.json', 'r') as f:
    endpoint_dets = json.loads(f.read())

def genereate_text(message, system_prompt="", max_tokens=1500):
    messages = []
    if system_prompt:
        messages.append({
                    "role": "system",
                    "content": system_prompt
                })
    messages.append({
                    "role": "user",
                    "content": message
                })
    request_instance = {
            "@requestFormat": "chatCompletions",
            "messages": messages,
            "max_tokens": max_tokens
        }
    result = predict_custom_trained_model_sample(
                **endpoint_dets,
                instances=request_instance,
            )
    return result.predictions[0]['choices'][0]['message']['content'].strip()

def generate_specialized_text(system_prompt, max_toxens=1500):
    def generate_text_spec(message):
        return genereate_text(message, system_prompt=system_prompt, max_tokens=max_toxens)
    return generate_text_spec
