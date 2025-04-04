#!/usr/bin/python3
import sys
import json

json_obj = json.load(sys.stdin)
json_obj["schema_id"] = 33

# sys.argv[1] <-- this is how you access arguments passed in from flowfile attributes

print(json_obj["schema_id"])
