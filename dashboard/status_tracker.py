STEPS = {
    1: "Pending",
    2: "Pending",
    3: "Protected",
    4: "Protected",
    5: "Pending",
    6: "Pending"
}

def update_step(step, status):
    STEPS[step] = status

def get_steps():
    return STEPS