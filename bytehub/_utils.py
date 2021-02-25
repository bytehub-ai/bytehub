import json
import base64
import cloudpickle


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def serialize(func):
    return base64.b64encode(cloudpickle.dumps(func)).decode("utf-8")


def deserialize(func_string):
    return cloudpickle.loads(base64.b64decode(func_string.encode("utf-8")))
