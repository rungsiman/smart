import json
import re


def parse_kwargs(kwargs_list):
    kwargs_dict = {}

    for kwarg in kwargs_list:
        value = '='.join(kwarg.split('=')[1:])

        if re.match(r'^(-)?[0-9]+$', value):
            value = int(value)

        elif re.match(r'^(-)?[0-9]*.[0-9]+$', value) or re.match(r'^(-)?[0-9]*(\.)?[0-9]+e(-|\+)[0-9]+$', value):
            value = float(value)

        elif re.match(r'^\[.*]$', value):
            value = json.loads(value)

        elif value.lower() in ('true', 'false'):
            value = value == 'true'

        elif value.lower() == 'none':
            value = None

        kwargs_dict[kwarg[2:].split('=')[0]] = value

    return kwargs_dict
