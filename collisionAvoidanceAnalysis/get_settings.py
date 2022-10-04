import json

def get_settings(json_name="settings.json"):
    with open(json_name, "r") as file:
        settings = json.load(file)
    return settings