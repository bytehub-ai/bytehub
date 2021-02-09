import json


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False
