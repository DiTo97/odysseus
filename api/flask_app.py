import json
import typing as t
import zmq

from absl import logging

from flask import Flask
from flask import request

# Custom imports
from api_request import Type

from server import _Request
from server import _Server


MAX_displacement = 3
MAX_vehicles = 400


_Server['ENDPOINT'] = '{}://{}:{}'.format(_Server['PROTOCOL'],
                                          _Server['HOST'], _Server['PORT'])


ESBDQN_keys = [
    'P_zone', 'D_zone'
]

ESBDQN_keys_opt = [
    'P_zone_sugg',
    'D_zone_sugg',
]


Methods = {
    'ALL': ['DELETE', 'GET',  'HEAD',
            'PATCH',  'POST', 'PUT' ],
}


app = Flask(__name__)


def valid_request(methods: t.List[str],
                  _request: t.Any) \
                 -> t.Tuple[bool, t.Optional[
                        t.Tuple[t.Dict, int]]]:
    if _request.method \
            not in methods:
        return False, ({
                   'error': 'Request method not allowed'
                       }, 405)

    scenario_data = _request.json

    scenario_keys = scenario_data.keys() \
        if scenario_data is not None else []

    if not set(ESBDQN_keys).issubset(
            set(scenario_keys)):
        return False, ({
                   'error': 'Missing parameters'
                       }, 400)

    if not set(scenario_keys).issubset(
            set(ESBDQN_keys) | set(ESBDQN_keys_opt)):
        return False, ({
                   'error': 'Unknown parameters'
                       }, 400)

    return True, None


def send_request(_data: t.Any, _type: Type) \
                   -> t.Optional[t.Dict]:
    try:
        with zmq.Context() as context:
            logging.info("Connecting to server…")

            _request = json.dumps({
                'data': _data,
                'type': _type
            }).encode()

            client = context.socket(zmq.REQ)
            client.connect(_Server['ENDPOINT'])

            client.send(_request)

            status = client.poll(_Request['TIMEOUT'])

            if (status & zmq.POLLIN) != 0:
                logging.info('Got reply from server')

                response = client.recv()
                response = response.decode()

                # Socket is confused by the NOP.
                # Close and remove it.
                client.setsockopt(zmq.LINGER, 0)
                client.close()

                return json.loads(response)

            # Socket is confused by the NOP.
            # Close and remove it.
            client.setsockopt(zmq.LINGER, 0)
            client.close()

            logging.error("Server seems to be offline."
                          " Abandoning...")
            return None
    except Exception:
        pass


def send_response(response: t.Optional[t.Dict],
                  json_ser_keys: t.Optional[
                      t.List[str]] = None):
    if response is None:
        return {
            'error': 'No response from server'
               }, 503

    if json_ser_keys is not None:
        # Deserialize response data
        for k in json_ser_keys:
            if k in response.keys():
                response[k] = \
                    json.loads(response[k])

    if 'error' in response.keys():
        return {
            'error': response['error']
               }, 500

    return response, 200


@app.route('/suggest-pick-up', methods=Methods['ALL'])
def suggest_pick_up():
    status, error = valid_request(['POST'], request)

    if not status:
        return error

    response = send_request(request.json,
                            Type.Pick_up)

    return send_response(response, ['vehicle'])


@app.route('/suggest-drop-off', methods=Methods['ALL'])
def suggest_drop_off():
    status, error = valid_request(['POST'], request)

    if not status:
        return error

    response = send_request(request.json,
                            Type.Drop_off)

    return send_response(response)


@app.route('/get-incentive', methods=Methods['ALL'])
def get_incentive():
    if request.method \
            not in ['GET']:
        return {
            'error': 'Request method not allowed'
               }, 405

    scenario_data = request.args

    scenario_keys = scenario_data.keys() \
        if scenario_data is not None else []

    if not ({'P_zone', 'P_zone_sugg'}
                .issubset(set(scenario_keys))
            or {'D_zone', 'D_zone_sugg'}
                .issubset(set(scenario_keys))):
        return {
            'error': 'Missing parameters'
               }, 400

    if not {'P_zone', 'P_zone_sugg'}       \
             .issubset(set(scenario_keys)) \
        and ('P_zone' in scenario_keys
             or 'P_zone_sugg' in scenario_keys):
        return {
            'error': 'Missing pick-up parameters'
               }, 400

    if not {'D_zone', 'D_zone_sugg'}       \
             .issubset(set(scenario_keys)) \
        and ('D_zone' in scenario_keys
             or 'D_zone_sugg' in scenario_keys):
        return {
            'error': 'Missing drop-off parameters'
               }, 400

    if not set(scenario_keys).issubset(
            set(ESBDQN_keys) | set(ESBDQN_keys_opt)):
        return {
            'error': 'Unknown parameters'
               }, 400

    response = send_request(request.args,
                            Type.Incentive)

    return send_response(response)


if __name__ == '__main__':
    app.run(host='127.0.0.1',
            port=8000, debug=True)
