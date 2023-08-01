import logging

from flask import Flask, request
from flask_cors import CORS, cross_origin
import pandas as pd
import pm4py.objects.bpmn.importer.variants.lxml as bpmn_importer
from io import StringIO

from process_patching_integration import repair_process_model

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

formatter = logging.Formatter(  # pylint: disable=invalid-name
    '%(asctime)s %(levelname)s %(process)d ---- %(threadName)s  '
    '%(module)s : %(funcName)s {%(pathname)s:%(lineno)d} %(message)s', '%Y-%m-%dT%H:%M:%SZ')

handler = logging.StreamHandler()
handler.setFormatter(formatter)

app.logger.setLevel(logging.DEBUG)
app.logger.addHandler(handler)


@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())


'''
This end point is used to repair process models with new events
The process model is identified by the process_id
Processmodel and event log and activity mappings are send in the JSON request body in separate fields
'''


@app.route('/api/v1/patchModel', methods=['POST'])
@cross_origin()
def patch_model():
    app.logger.debug("Patch Model request received on python worker")
    app.logger.debug("Try event Log retrival...")
    # get event log from request body
    try:
        event_log = pd.read_csv(StringIO(request.json['event_log_csv']), sep=',')
        app.logger.debug("Event Log retrival done: " + str(event_log.shape))
    except Exception as e:
        app.logger.debug("Error while reading event log csv: " + str(e))


    app.logger.info("Try process model retrival...")
    # get process model from request body
    bpmn = bpmn_importer.import_from_string(request.json['process_model']["xmlModel"])
    app.logger.info("Process model retrival should be done.")

    app.logger.info("Try activity mappings retrival...")
    # get activity mappings from request body
    activity_mappings = request.json['process_model']['activityMappings']
    app.logger.info("Activity mappings retrival done. Count Activity Mappings: " + str(len(activity_mappings)))

    app.logger.info("Try process model repair...")
    changeset, bpmn_repaired = repair_process_model(bpmn, event_log, activity_mappings)
    app.logger.info("Process model repair done. Changeset length: " + str(len(changeset)))

    '''"id": request.json['process_model']['id'],
        "title": request.json['process_model']['title'],
        "descriptionShort": request.json['process_model']['descriptionShort'],
        "description": request.json['process_model']['description'],
        "xmlModel": request.json['process_model']["xmlModel"],'''
    return {
        "changeSet": changeset,
        "repairedModel": bpmn_repaired
    }


if __name__ == '__main__':
    app.run()
