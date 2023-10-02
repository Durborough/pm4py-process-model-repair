import uuid
import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.utils.petri_utils import get_transition_by_name
from pm4py.objects.petri_net.utils.final_marking import discover_final_marking
from pm4py.objects.petri_net.utils.initial_marking import discover_initial_marking
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.conformance.tokenreplay import algorithm as token_based_replay
from pm4py.objects.bpmn.layout.variants.graphviz import Parameters as bpmn_layouter_parameters
from pm4py.objects.bpmn.layout import layouter as bpmn_layouter
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter


def get_place_by_name(net: PetriNet, place_name):
    for p in net.places:
        if p.name == place_name:
            return p
    return None


# Get all the traces from a given log as a dictionary
# Key is the trace name
# Value is the corresponding log snippet sorted by time stamp
def get_traces_from_log(log):
    traces = {}
    for case_id in log['case:concept:name'].unique():
        trace = log[log['case:concept:name'] == case_id]
        trace = trace.sort_values(by=['time:timestamp'])
        traces[case_id] = trace
    return traces


# Returns true if the given trace fits the given process model
def trace_is_fit(trace, net, im, fm):
    fitness = pm4py.conformance.fitness_alignments(trace, net, im, fm)
    return fitness['averageFitness'] == 1.0


def get_reached_marking(trace, net, im, fm):
    parameters_tbr = {token_based_replay.Variants.TOKEN_REPLAY.value.Parameters.STOP_IMMEDIATELY_UNFIT: True}
    fitness = token_based_replay.apply(trace, net, im, fm, parameters=parameters_tbr)
    return fitness[0]['reached_marking']


# Returns the alignment between the given trace and the given process model as a list of tuples
def get_alignment(trace, net, im, fm):
    alignment = pm4py.conformance_diagnostics_alignments(trace,
                                                         net,
                                                         im,
                                                         fm,
                                                         variant_str=alignments.VERSION_DISCOUNTED_A_STAR)
    return alignment[0]['alignment']


# Returns true if the given move is a synchronous move
def is_sync_move(move):
    return move[0] == move[1] and move[0] != '>>' and move[1] != '>>'


# Returns true if the given move is a move on log
def is_move_on_log(move):
    return move[0] != move[1] and move[0] != '>>' and move[1] == '>>'


# Returns true if the given move is a move on model
def is_move_on_model(move):
    return move[0] != move[1] and move[0] == '>>' and move[1] != '>>'


# Returns true if the given move is a move on model and involves a skip / hidden transition
def is_move_on_model_skip(move):
    return is_move_on_model(move) and move[1] is None


# Returns net with an additional skip transition connected to the given in_places and out_places
# The skip transition's name should be unique
def add_skip_transition(net, in_place, out_place, skip_transition_name):
    petri_utils.add_transition(net, name=skip_transition_name, label=None)
    new_from_transition = get_transition_by_name(net, skip_transition_name)
    petri_utils.add_arc_from_to(in_place, new_from_transition, net)
    petri_utils.add_arc_from_to(new_from_transition, out_place, net)
    return net


def get_transition_by_label(net: PetriNet, transition_label):
    for t in net.transitions:
        if t.label == transition_label:
            return t
    return None


# Input: net and a list of names of transitions to be skipped in that net
# Output: net with the transitions skipped by skip transitions with same in_places and out_places as skipped transitions
def skip_transition(net, transition_to_be_skipped_name):
    SKIP_NODE_BASE_NAME = "skip_transition_skip_BASE_NAME_"

    transition_to_be_skipped = get_transition_by_label(net, transition_to_be_skipped_name)
    in_arcs = transition_to_be_skipped.in_arcs
    out_arcs = transition_to_be_skipped.out_arcs
    # The SKIP_NODE_BASE_NAME + transition is a unique name since transitions to be skipped is a set
    petri_utils.add_transition(net, name=SKIP_NODE_BASE_NAME + str(transition_to_be_skipped_name), label=None)
    new_transition = get_transition_by_name(net, SKIP_NODE_BASE_NAME + str(transition_to_be_skipped_name))
    for arc in in_arcs:
        petri_utils.add_arc_from_to(arc.source, new_transition, net)
    for arc in out_arcs:
        petri_utils.add_arc_from_to(new_transition, arc.target, net)
    return net


# Input: Dictionary of unfit traces where trace name is the key and trace is the value,
# petri net wit initial and final markings
# Output: Set of transitions to be skipped to repair moves on model
def get_transitions_to_be_skipped(traces_to_be_repaired, net, im, fm):
    transitions_to_be_skipped = set()
    for trace_key, trace in traces_to_be_repaired.items():
        # Calculate the alignment between the trace and the process model
        alignment = get_alignment(trace, net, im, fm)

        # If it is a move on model (e.g.: (>>, a)), add the skipped transition to a set of transitions to be skipped
        # Ignore synchronous moves (e.g.: (a, a)), and moves on log (e.g.: (a, >>)) for now
        transitions_to_be_skipped.update(
            set(map(lambda y: y[1], filter(lambda x: x[0] == '>>' and x[1] != '>>', alignment))))
    return transitions_to_be_skipped


# Create map of last place reached as key and sub-log dataframe of sub-traces as value
def get_sublogs_with_places(traces_to_be_repaired, event_log, net, im, fm):
    sublogs = {}

    for trace_key, trace in traces_to_be_repaired.items():
        # Continue with next trace, if model is already repaired for current trace based on token based replay
        if trace_is_fit(trace, net, im, fm):
            continue

        # The move_on_log_sub-trace dataframe temporarily stores the sub-trace of moves on log
        # before a synchronous move is reached
        move_on_log_subtrace = pd.DataFrame(columns=event_log.columns)
        sync_moves_subtrace = pd.DataFrame(columns=event_log.columns)

        # Store the last markings reached initialised with the start place of the petri net.
        # This helps with not needing to do a full token based replay for each sub-trace.
        last_reached_marking = im

        # Identify all sub-traces of moves on log using the alignment and save them in the sub-traces map
        alignment = get_alignment(trace, net, im, fm)

        # Last item fix
        alignment_len = len(alignment) - 1
        for index, element in enumerate(alignment):
            # If it is a synchronous move (e.g.: (a, a)):

            if is_sync_move(element):
                # If move_on_log_subtrace is not empty, this means, that the current alignment step ends a sub-trace,
                # thus save the sub-trace to the sub-traces map with the last place reached as key
                if not move_on_log_subtrace.empty:
                    # Get last reached marking of the sub-trace.
                    # This corresponds to the marking reached before the sub-trace of moves on log started
                    last_reached_marking = get_reached_marking(sync_moves_subtrace, net, im, fm)

                    # save the sub-trace of moves on log in the sub-traces map with the last place reached as key
                    sublogs[last_reached_marking] = pd.concat(
                        [sublogs.get(last_reached_marking, pd.DataFrame(columns=event_log.columns)),
                         move_on_log_subtrace], ignore_index=True)

                    # Reset the move_on_log_subtrace to be ready for the next potential sub-trace
                    move_on_log_subtrace = pd.DataFrame(columns=event_log.columns)
                # Drop first activity in event log
                # (this should correspond to the sync move in the alignment element currently at hand) &
                # continue with the next element
                sync_moves_subtrace = pd.concat([sync_moves_subtrace, trace.iloc[0].to_frame().T], ignore_index=True)
                trace = trace.iloc[1:]

            # If it is a move on log (e.g.: (a, >>)):
            elif is_move_on_log(element):
                # append it to the dataframe of moves on log and remove it from the trace
                # to_frame().T transforms the series resulting form the iloc[0] operation to a dataframe with one row
                move_on_log_subtrace = pd.concat([move_on_log_subtrace, trace.iloc[0].to_frame().T], ignore_index=True)
                trace = trace.iloc[1:]

                if index == alignment_len:
                    # Get last reached marking of the sub-trace.
                    # This corresponds to the marking reached before the sub-trace of moves on log started
                    last_reached_marking = get_reached_marking(sync_moves_subtrace, net, im, fm)

                    # save the sub-trace of moves on log in the sub-traces map with the last place reached as key
                    sublogs[last_reached_marking] = pd.concat(
                        [sublogs.get(last_reached_marking, pd.DataFrame(columns=event_log.columns)),
                         move_on_log_subtrace], ignore_index=True)

                    # Reset the move_on_log_subtrace to be ready for the next potential sub-trace
                    move_on_log_subtrace = pd.DataFrame(columns=event_log.columns)


            # Ignore all other alignment elements (e.g.: (>>, a)), since they have been fixed in previous steps
            else:
                if index == alignment_len:
                    # Get last reached marking of the sub-trace.
                    # This corresponds to the marking reached before the sub-trace of moves on log started
                    last_reached_marking = get_reached_marking(sync_moves_subtrace, net, im, fm)

                    # save the sub-trace of moves on log in the sub-traces map with the last place reached as key
                    sublogs[last_reached_marking] = pd.concat(
                        [sublogs.get(last_reached_marking, pd.DataFrame(columns=event_log.columns)),
                         move_on_log_subtrace], ignore_index=True)

                    # Reset the move_on_log_subtrace to be ready for the next potential sub-trace
                    move_on_log_subtrace = pd.DataFrame(columns=event_log.columns)
                continue
    return sublogs


# Greedy algorithm to optimize the sublogs
# Input: sublogs in the form of a dictionary with the key being markings and the value being corresponding sublogs
# Output: optimized sublogs in the form of a dictionary with the key being a place and the value being sublog
#         The sub-log is optimized in the sense that the places are disjoint,
#         meaning the net can be fixed in as few places as possible, while the sublogs are still complete
def optimize_sublogs(sublogs):
    # Create empty dictionary to store optimized sublogs
    optimized_sublogs = {}

    while len(sublogs) > 0:
        # Create empty dictionary to store place count as value and place as key
        place_count = {}
        # get all keys from sublogs
        keys = list(sublogs.keys())
        # Flatten the keys to a list of places
        places = [item for sublist in keys for item in sublist]

        # Count the occurrences of each place in the list of places
        for place in places:
            place_count[place] = place_count.get(place, 0) + 1

        # get key of maximum count
        max_count_place = max(place_count, key=place_count.get)
        # List of keys to delete
        keys_to_delete = []
        # Union all values of sublogs with max_count_place in key and
        # store in optimized_sublogs with max_count_place as key
        for key, value in sublogs.items():
            if max_count_place in key:
                optimized_sublogs[max_count_place] = pd.concat(
                    [optimized_sublogs.get(max_count_place, pd.DataFrame(columns=value.columns)), value],
                    ignore_index=True)
                # mark key for deletion
                keys_to_delete.append(key)
        # delete all marked keys from sublogs
        for key in keys_to_delete:
            del sublogs[key]

    return optimized_sublogs


# Input: BPMN Diagram
# Output: BPMN diagram but all Start and End-events have been removed
def remove_start_and_end_events_from_bpmn(bpmn_graph):
    outgoing_edges = {}
    incoming_edges = {}

    for flow in bpmn_graph.get_flows():
        source = flow.get_source()
        target = flow.get_target()

        if source not in outgoing_edges:
            outgoing_edges[source] = set()
            outgoing_edges[source].add(flow)

        if target not in incoming_edges:
            incoming_edges[target] = set()
            incoming_edges[target].add(flow)

    nodes = list(bpmn_graph.get_nodes())
    for node in nodes:
        if isinstance(node, pm4py.BPMN.StartEvent):
            for flow in outgoing_edges[node]:
                if flow in bpmn_graph.get_flows():
                    bpmn_graph.remove_flow(flow)
            bpmn_graph.remove_node(node)
        if isinstance(node, pm4py.BPMN.EndEvent):
            for flow in incoming_edges[node]:
                if flow in bpmn_graph.get_flows():
                    bpmn_graph.remove_flow(flow)
            bpmn_graph.remove_node(node)
    return bpmn_graph


def merge_and_connect_at_place(net, im, fm, sub_net, sub_im, sub_fm, place):
    SUBPROCESS_BASE_NAME = "subprocess_"
    # Merge & connect the sub-net to the original petri net
    net = petri_utils.merge(net, [sub_net])
    # Extract the start place and final of the sub-process net. This assumes that there is only on of each
    sub_start_place = list(sub_im.keys())[0]
    sub_final_place = list(sub_fm.keys())[0]
    # Connect the sub_net with the net at the place
    net = add_skip_transition(net, place, sub_start_place, SUBPROCESS_BASE_NAME + '_from_' + place.name)
    net = add_skip_transition(net, sub_final_place, place, SUBPROCESS_BASE_NAME + '_to_' + place.name)
    return net, im, fm
'''
def merge_and_connect_at_place(net, im, fm, sub_net, sub_im, sub_fm, place):
    SUBPROCESS_BASE_NAME = "subprocess_"
    # Merge & connect the sub-net to the original petri net
    net = petri_utils.merge(net, [sub_net])
    # Extract the start place and final of the sub-process net. This assumes that there is only on of each
    sub_start_place = list(sub_im.keys())[0]
    sub_final_place = list(sub_fm.keys())[0]

    # split up place
    double_place = PetriNet.Place(name=place.name + "_double")
    net.places.add(double_place)
    for in_arc in place.in_arcs.copy():
        petri_utils.add_arc_from_to(in_arc.source, double_place, net)
        petri_utils.remove_arc(net, in_arc)
    net = add_skip_transition(net, double_place, place, "split_" + place.name)


    # Connect the sub_net with the net at the place
    net = add_skip_transition(net, double_place, sub_start_place, SUBPROCESS_BASE_NAME + '_from_' + place.name)
    net = add_skip_transition(net, sub_final_place, place, SUBPROCESS_BASE_NAME + '_to_' + place.name)
    return net, im, fm
'''

# Input: List of BPMN Transitions preceding and following the change as well as a bpmn_graph, that shows the change
# as well as the immediate surroundings of the change Output: Dictionary conforming to the "change" API
def create_changeset_entry(transitions_preceding, transitions_following, bpmn_graph, change_type):
    if change_type == "Skip Activity":
        description = "Make activities optional"
    elif change_type == "Additional Activity":
        description = "Allow additional behavior"
    else:
        description = ""
    change = {"id": str(uuid.uuid4()),
              "change": {"type": change_type, "description": description},
              "src": [transition.label for transition in transitions_preceding],
              "target": [transition.label for transition in transitions_following],
              "proposedChange": bpmn_exporter.serialize(bpmn_graph).decode('utf-8')}
    return change


# Input: Petri Net, two places and a transition, that should connect them
# Output: Petri net with the two places connected via a copy of the transition
def connect_places_via_transition(net, transition, from_place, to_place):
    petri_utils.add_transition(net, name=transition.name, label=transition.label)
    added_transition = get_transition_by_name(net, transition.name)
    petri_utils.add_arc_from_to(from_place, added_transition, net)
    petri_utils.add_arc_from_to(added_transition, to_place, net)
    return net


# Extract one place from a given marking
def extract_place_from_marking(marking):
    return list(marking.keys())[0]


# Adds the surroundings of a place to a given sub-net (net, im, fm)
def add_surroundings(net, im, fm, place):
    sub_start_place = extract_place_from_marking(im)
    sub_final_place = extract_place_from_marking(fm)
    new_start_place = PetriNet.Place(name='new_start_place')
    net.places.add(new_start_place)
    new_final_place = PetriNet.Place(name='new_final_place')
    net.places.add(new_final_place)
    # get the incoming / outgoing arcs of the place and identify the transitions producing to this place
    transitions_producing_to_place = [arc.source for arc in place.in_arcs]
    transitions_consuming_from_place = [arc.target for arc in place.out_arcs]
    # TODO: think about hidden transitions
    # Add the transitions connected to the place before and after the newly discovered section
    for transition in transitions_producing_to_place:
        if "SKIP_Transition_" in transition.name:
            continue
        if "SKIP_Place_" in transition.name:
            continue
        if "skip_transition_skip_BASE_NAME_" in transition.name:
            continue

        net = connect_places_via_transition(net, transition, new_start_place, sub_start_place)
    for transition in transitions_consuming_from_place:
        if "SKIP_Transition_" in transition.name:
            continue
        if "SKIP_Place_" in transition.name:
            continue
        if "skip_transition_skip_BASE_NAME_" in transition.name:
            continue
        net = connect_places_via_transition(net, transition, sub_final_place, new_final_place)
    return net, im, fm, transitions_producing_to_place, transitions_consuming_from_place


# Adds a skip transition between a place of the initial marking and final marking, thus making the whole process
# skippable
def make_process_skippable(net, im, fm):
    start_place = extract_place_from_marking(im)
    final_place = extract_place_from_marking(fm)
    net = add_skip_transition(net, start_place, final_place, "SKIP_Place_" + str(start_place))
    im = discover_initial_marking(net)
    fm = discover_final_marking(net)
    return net, im, fm


# Input: petri net
# Output: petri net with all transition names (ids) in the labels and the mapping of the names (ids) to the labels
def preprocess_petri_net(net, activity_mappings):
    activity_id_to_label = {}
    for transition in net.transitions:
        if transition.label == None:
            continue
        activity_id_to_label[activity_mappings[transition.name]] = transition.label
        transition.label = activity_mappings[transition.name]
    return net, activity_id_to_label


# Input: eventlog in Dataframe form
# Output: Dataframe with registered Activity ID as key, and corresponding Names
def create_translation_dataframe(df):
    translation_df = df.set_index('registeredActivity')
    translation_df = translation_df[~translation_df.index.duplicated(keep='first')]
    return translation_df[['activityName']]


def preprocess_net_and_event_log(net, raw_event_log, activity_mappings):
    # Format
    formated_event_log = pm4py.format_dataframe(raw_event_log, case_id='processInstanceId',
                                                activity_key='registeredActivity', timestamp_key='timestamp')
    # Filter The Input Log should already just contain events from one processId. Filter for processId, to be sure
    # the log is clean.
    filtered_event_log = formated_event_log[
        (formated_event_log['transition'] == 'TERMINATE')]  # (formated_event_log['processId'] == process_id) &

    translations_id_to_label = create_translation_dataframe(formated_event_log)

    net, activity_id_to_label = preprocess_petri_net(net, activity_mappings)
    for activity_id, label in activity_id_to_label.items():
        translations_id_to_label.loc[activity_id] = label
    return net, filtered_event_log, translations_id_to_label


# Input: A net discovered from the event log and a translation_df with registeredActivityIds as index.
# The process discovery step will result in random uuids as transition names and the registeredActivityId as the label.
# To make it human-readable, the correct activity name is looked up in the translation_df and set as the transition name
#
def fix_transition_naming(net, translation_df):
    for transition in net.transitions:
        if transition.label is not None:
            transition_label_old = transition.label
            transition.label = translation_df.loc[transition_label_old]['activityName']
            transition.name = transition_label_old

    return net


# This Algorithm implements the process model repair Algorithm
# inspired by the paper "Repairing process models to reflect reality" by Dirk Fahland and Wil van der Aalst
# Input: bpmn model, event log, activity mappings
# Output: repaired bpmn process model, change set
def repair_process_model(bpmn, event_log, activity_mappings):
    # transform process model into petri net
    # This retains the activity ids as names of the transitions
    net, im, fm = pm4py.convert_to_petri_net(bpmn)

    # Preprocess the petri net, by putting all the transition names (ids) in the labels.
    # At the same time, save the mapping of the names (ids) to the labels
    # This is necessary since transition labels might not be unique and
    # process patching step involves check with the activity ids

    net, event_log, translations_id_to_label = preprocess_net_and_event_log(net, event_log, activity_mappings)
    # net, activity_id_to_label = preprocess_petri_net(net, activity_mappings)

    # Extract the different traces from the event log
    traces = get_traces_from_log(event_log)

    # Identify the traces that are not fitting the process model:
    traces_to_be_repaired = {k: v for k, v in traces.items() if not trace_is_fit(v, net, im, fm)}

    # 1. Fix all moves on model
    # Identify the transitions to be skipped
    transitions_to_be_skipped = set()
    # Identify the transitions to be skipped
    transitions_to_be_skipped = get_transitions_to_be_skipped(traces_to_be_repaired, net, im, fm)

    # Skip all transitions to be skipped by adding a new invisible "None" transition to the process model
    # with the same input and output places as the transition to be skipped
    for transition in transitions_to_be_skipped:
        skip_transition(net, transition)

    # Alignment of net and traces now do not contain any moves on model besides moves on hidden (None) transitions

    # 2. Fix all moves on log
    # Create map of last place reached as key and sub-log dataframe of sub-traces as value
    sublogs = get_sublogs_with_places(traces_to_be_repaired, event_log, net, im, fm)
    # 3. Optimise the sublogs
    sublogs = optimize_sublogs(sublogs)

    changeset = []

    # Add changes to changeset for moves on model
    for transition_name in transitions_to_be_skipped:
        sub_net = pm4py.PetriNet()

        transition = get_transition_by_label(net, transition_name)

        # Add transition to be skipped
        petri_utils.add_transition(sub_net, name=transition.name, label=transition.label)
        sub_transition_to_be_skipped = get_transition_by_name(sub_net, transition.name)
        # Add skip of this transition
        skip_place_in = PetriNet.Place(name='skip_place_in')
        sub_net.places.add(skip_place_in)
        skip_place_out = PetriNet.Place(name='skip_place_out')
        sub_net.places.add(skip_place_out)
        petri_utils.add_arc_from_to(skip_place_in, sub_transition_to_be_skipped, sub_net)
        petri_utils.add_arc_from_to(sub_transition_to_be_skipped, skip_place_out, sub_net)
        sub_net = add_skip_transition(sub_net, skip_place_in, skip_place_out,
                                      "SKIP_Transition_" + str(sub_transition_to_be_skipped))

        # Identify places before and after
        in_places = [arc.source for arc in transition.in_arcs]
        out_places = [arc.target for arc in transition.out_arcs]

        # Identify transitions for these in and out places

        transitions_preceeding = []
        transitions_preceeding_in_net = []

        for place in in_places:
            sub_place = PetriNet.Place(name=place.name)
            sub_net.places.add(sub_place)
            sub_net = add_skip_transition(sub_net, sub_place, skip_place_in, "SKIP_Place_in_" + str(sub_place))

            transitions_producing_to_place = [arc.source for arc in place.in_arcs]
            for trans in transitions_producing_to_place:
                if "SKIP_Transition_" in trans.name:
                    continue
                if "SKIP_Place_" in trans.name:
                    continue
                if "skip_transition_skip_BASE_NAME_" in trans.name:
                    continue

                petri_utils.add_transition(sub_net, name=trans.name, label=trans.label)
                sub_transition = get_transition_by_name(sub_net, trans.name)
                petri_utils.add_arc_from_to(sub_transition, sub_place, sub_net)
                transitions_preceeding.append(sub_transition)
                transitions_preceeding_in_net.append(trans)

        transitions_following = []
        transitions_following_in_net = []
        # Identify transitions for these in and out places

        for place in out_places:
            sub_out_place = PetriNet.Place(name=place.name)
            sub_net.places.add(sub_out_place)
            sub_net = add_skip_transition(sub_net, skip_place_out, sub_out_place, "SKIP_Place_in_" + str(sub_out_place))

            transitions_consuming_from_place = [arc.target for arc in place.out_arcs]
            for trans in transitions_consuming_from_place:
                if "SKIP_Transition_" in trans.name:
                    continue
                if "SKIP_Place_" in trans.name:
                    continue
                if "skip_transition_skip_BASE_NAME_" in trans.name:
                    continue
                # account for loops
                if trans.name in [transition.name for transition in sub_net.transitions]:
                    sub_transition = get_transition_by_name(sub_net, trans.name)
                else:
                    petri_utils.add_transition(sub_net, name=trans.name, label=trans.label)
                    sub_transition = get_transition_by_name(sub_net, trans.name)
                petri_utils.add_arc_from_to(sub_out_place, sub_transition, sub_net)
                transitions_following.append(sub_transition)
                transitions_following_in_net.append(trans)

        # Create places as new initial and final places
        sub_new_initial_place = PetriNet.Place(name="new_initial_place")
        sub_net.places.add(sub_new_initial_place)
        sub_new_final_place = PetriNet.Place(name="new_final_place")
        sub_net.places.add(sub_new_final_place)

        for sub_transition in transitions_preceeding:
            # Account for loops
            if sub_transition not in transitions_following:
                petri_utils.add_arc_from_to(sub_new_initial_place, sub_transition, sub_net)
        for sub_transition in transitions_following:
            # Account for loops
            if sub_transition not in transitions_preceeding:
                petri_utils.add_arc_from_to(sub_transition, sub_new_final_place, sub_net)

        sub_im = discover_initial_marking(sub_net)
        sub_fm = discover_final_marking(sub_net)

        # Convert to BPMN, bring into correct shape, add a layout and serialize

        sub_net = fix_transition_naming(sub_net, translations_id_to_label)

        sub_bpmn_graph = pm4py.convert_to_bpmn(sub_net, sub_im, sub_fm)
        # sub_bpmn_graph = remove_start_and_end_events_from_bpmn(sub_bpmn_graph)
        sub_bpmn_graph_layouted = bpmn_layouter.apply(sub_bpmn_graph,
                                                      parameters={
                                                          bpmn_layouter_parameters.SCALING_FACTOR: 1.0,
                                                          bpmn_layouter_parameters.SCREEN_SIZE_X: 1400,
                                                          bpmn_layouter_parameters.SCREEN_SIZE_Y: 800
                                                      })

        # sub_bpmn_graph_layouted = remove_start_and_end_events_from_bpmn(sub_bpmn_graph_layouted)

        changeset.append(
            create_changeset_entry(transitions_preceeding_in_net, transitions_following_in_net, sub_bpmn_graph_layouted,
                                   "Skip Activity"))

    # Add changes to changeset for moves on log
    for place, sublog in sublogs.items():
        # Mine the sub-log for a bpmn model of the subprocess
        # Mine the sub-log for a process model petri net
        sub_net, sub_im, sub_fm = pm4py.discover_petri_net_inductive(sublog,
                                                                     activity_key='concept:name',
                                                                     case_id_key='case:concept:name',
                                                                     timestamp_key='time:timestamp')

        sub_net, sub_im, sub_fm, transitions_producing_to_place, transitions_consuming_from_place = add_surroundings(
            sub_net, sub_im, sub_fm, place)
        sub_net, sub_im, sub_fm = make_process_skippable(sub_net, sub_im, sub_fm)

        sub_net = fix_transition_naming(sub_net, translations_id_to_label)

        sub_bpmn_graph = pm4py.convert_to_bpmn(sub_net, sub_im, sub_fm)

        sub_bpmn_graph_layouted = bpmn_layouter.apply(sub_bpmn_graph,
                                                      parameters={
                                                          bpmn_layouter_parameters.SCALING_FACTOR: 1.0,
                                                          bpmn_layouter_parameters.SCREEN_SIZE_X: 1400,
                                                          bpmn_layouter_parameters.SCREEN_SIZE_Y: 800
                                                      })

        changeset.append(create_changeset_entry(transitions_producing_to_place, transitions_consuming_from_place,
                                                sub_bpmn_graph_layouted, "Additional Activity"))

    # 4. Repair the model for each sublog, by discovering the process model of the sublog and
    # merging it into the original petri net at the last place reached before the sublog started
    for place, sublog in sublogs.items():
        sub_net, sub_im, sub_fm = pm4py.discover_petri_net_inductive(sublog, activity_key='concept:name',
                                                                     case_id_key='case:concept:name',
                                                                     timestamp_key='time:timestamp')
        net, im, fm = merge_and_connect_at_place(net, im, fm, sub_net, sub_im, sub_fm, place)

    net = fix_transition_naming(net, translations_id_to_label)
    bpmn_graph = pm4py.convert_to_bpmn(net, im, fm)

    bpmn_graph_layouted = bpmn_layouter.apply(bpmn_graph,
                                              parameters={
                                                  bpmn_layouter_parameters.SCALING_FACTOR: 1.0,
                                                  bpmn_layouter_parameters.SCREEN_SIZE_X: 1400,
                                                  bpmn_layouter_parameters.SCREEN_SIZE_Y: 800
                                              })
    bpmn_to_return = bpmn_exporter.serialize(bpmn_graph_layouted).decode('utf-8')


    # 5. Return the repaired process model
    return changeset, bpmn_to_return
