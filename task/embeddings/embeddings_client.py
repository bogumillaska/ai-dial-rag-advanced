import json
from urllib import request, response

import requests
from task._constants import API_KEY

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


#TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    
    def __init__(self, deployment_name: str, api_key:str):
        self.deployment_name = deployment_name
        self.api_key = api_key
        self._endpoint = DIAL_EMBEDDINGS.format(
            model=deployment_name
        )
        
    def get_embeddings(self, 
                       inputs: str | list[str], 
                       dimensions: int,
                       print_request: bool = True,
                       print_response: bool = False,
        ) -> list[list[str]]:
        headers = {
            "api-key": self.api_key,
            "content-type": "application/json",
        }

        request_data = {
            "input": inputs,
            "dimensions": dimensions,
        }

        response = requests.post(
            url=self._endpoint,
            headers=headers,
            json=request_data,
            timeout=60
        )

        embeddings = []
        if response.status_code == 200:
            response_json = response.json()
            data_list = response_json.get("data")
            for data in data_list:
                embeddings.append(data.get("embedding"))
        else:
            print(f"Error: {response.status_code} {response.text}")
        return embeddings

# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
