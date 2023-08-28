from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import streamlit as st

def get_response(prompt):

    # Your PAT (Personal Access Token) can be found in the portal under Authentification
    PAT = '906eb260478642778e943dff45f66f3e'
    # Specify the correct user_id/app_id pairings
    # Since you're making inferences outside your app's scope
    USER_ID = 'meta'
    APP_ID = 'Llama-2'
    # Change these to whatever model and text URL you want to use
    MODEL_ID = 'llama2-7b-chat'
    MODEL_VERSION_ID = 'e52af5d6bc22445aa7a6761f327f7129'
    TEXT_FILE_URL = 'https://samples.clarifai.com/negative_sentence_12.txt'


    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', 'Key ' + PAT),)
    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,  
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=prompt
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )

    if response.status.code != status_code_pb2.SUCCESS:
        print(response.status)
        return "Error: " + response.status.description

    output = response.outputs[0]
    predicted_concepts = "\n".join(["%s %.2f" % (concept.name, concept.value) for concept in output.data.concepts])
    return predicted_concepts

# Example usage
user_input = st.text_input("Enter text:")
if user_input:
    model_response = get_response(user_input)
    st.write("Model's Response:", model_response)
