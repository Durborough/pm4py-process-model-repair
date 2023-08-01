This repository contains an implementation of a process model repair algorithm. It can be used to repair a process model to fit an event log.

## Setup
Required are `pm4py` and `graphviz`. Install them with:
```
!pip -q install pm4py
!apt -q install graphviz
```

The file `process_model_repair_algorith.py` contains the implementation for repairing a Petri net to fit an event log. It repairs the Petri net in place.

**Usage:**
```
event_log = pm4py.read_xes('example.xes')
net, im, fm = pm4py.convert_to_petri_net(pm4py.read_bpmn('example.bpmn'))
pm4py.view_petri_net(net, im, fm)
net_repaired, im_repaired, fm_repaired = repair_process_model(net, im, fm, event_log)
pm4py.view_petri_net(net_repaired, im_repaired, fm_repaired)
```

The file `process_model_repair_with_change_set.py` contains the implementation for repairing a BPMN model to fit an event log and obtain a change set. It is the the function used in the integration into the PAIS Customate.

**Usage:**
```
repair_process_model(bpmn, event_log, activity_mappings)
```
Activity Mappings map between ids in the BPMN models and the activity ids in the event log.

The file `app.py` contains the flask server implementation for the integration into the PAIS Customate.