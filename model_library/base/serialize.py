# pyright: basic
import copyreg

from xai_sdk.proto.v6.chat_pb2 import GetChatCompletionResponse


def _unpickle_GetChatCompletionResponse(data):
    from xai_sdk.proto.v6.chat_pb2 import GetChatCompletionResponse

    msg = GetChatCompletionResponse()
    msg.ParseFromString(data)
    return msg


def _pickle_GetChatCompletionResponse(msg):
    return (_unpickle_GetChatCompletionResponse, (msg.SerializeToString(),))


copyreg.pickle(GetChatCompletionResponse, _pickle_GetChatCompletionResponse)
