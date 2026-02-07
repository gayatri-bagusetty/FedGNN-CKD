import json
import os

STATE_FILE = "data/system_state.json"


def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "federated_round": 0,
            "participating_hospitals": []
        }
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


def increment_federated_round():
    state = load_state()
    state["federated_round"] += 1
    save_state(state)


def set_participating_hospitals(hospitals):
    state = load_state()
    state["participating_hospitals"] = hospitals
    save_state(state)