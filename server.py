#!flask/bin/python
from flask import Flask, jsonify
from flask import abort
from flask import request

import base64

import classification as classifier

app = Flask(__name__)

# The API of paths looks more complex (feature is not as important right now)
# -> check out DroneScanningPathSegment struct on master branch in swift repo
paths = [{'id': 1},
         {'id': 2}]

# not sure yet about this either
sessions = [{'id': 1, 'state': 'running'},
            {'id': 2, 'state': 'finished'}]

# TODO: API NEEDS A REWORK

@app.route('/')
def index():
    return "I eat the fish."


@app.route('/images', methods=['POST'])
def upload_image():
    name = request.args.get('name')
    session_id = request.args.get('name')
    img = request.args.get('img')
    pos = request.args.get('pos')

    img = base64.b64decode(img)

    with open(name, 'wb') as f:
        f.write(img)
    
    # TODO: process image data (this should be done first)
    result = classifier.classify(name)
    print('Classification result: ' + str(result[0]) + ' - ' + str(result[1]))

    return str(result[0])


@app.route('/sessions', methods=['POST'])
def start():
    session = {
        'id': sessions[-1]['id'] + 1,
        'state': 'running'
    }
    sessions.append(session)
    return jsonify({'session': session}), 201


@app.route('/sessions/<int:session_id>', methods=['PUT'])
def stop(session_id):
    session = [item for item in sessions if item['id'] == session_id]
    session[0]['state'] = 'finished'
    print(sessions)
    return jsonify({'session': session}), 201


@app.route('/flightpaths', methods=['GET'])
def get_flight_paths():
    return jsonify({'paths': paths})


@app.route('/flightpaths/<int:path_id>', methods=['GET'])
def get_flight_path(path_id):
    path = [path for path in paths if path['id'] == path_id]
    if len(path) == 0:
        abort(404)
    return jsonify({'path': path[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
