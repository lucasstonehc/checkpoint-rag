import os
import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth

load_dotenv()

def create_weaviate_client():
    url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "WEAVIATE_URL and WEAVIATE_API_KEY must be configured"
        )

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
    )

    if not client.is_ready():
        client.close()
        raise ConnectionError("Weaviate is not ready")

    return client