import requests
import json
import conf.key


def post(txt):
    """post text to slack

    Args:
        txt ([str]):
    """

    webhook_url = conf.key.webhook_url
    requests.post(webhook_url, data=json.dumps({"text": txt}))
