import numpy as np


class NFA:

    def __init__(self, transition_matrix: list[list[float]], state_names: list[str], starting_heur: str):
        # self.all_state = set(states.keys())
        self.transition_matrix = np.array(transition_matrix)
        self.state_names = state_names
        self.states = np.arange(len(state_names))
        # self.states = list[range(len(state_names))]

        assert len(state_names) > 1
        assert len(state_names) == len(transition_matrix) and len(state_names) == len(transition_matrix[0])

        self.current_state = state_names.index(starting_heur)

    def get_transition_matrix(self):
        return self.transition_matrix.copy()

    def get_current_state(self):
        return self.state_names[self.current_state]

    def random_step(self):
        self.current_state = np.random.choice(self.states, p=self.transition_matrix[self.current_state])
        return self.state_names[self.current_state]


def main():
    transition_matrix = [
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.1, 0.6, 0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ]

    state_names = ["size", "max_dist", "avg_dist", "med_dist", "var_dist"]

    nfa = NFA(transition_matrix, state_names)

    counters = [1, 0, 0, 0, 0]

    for i in range(100_000):
        state = nfa.random_step()
        counters[state] += 1
        # print(state)
    print()
    print(counters)
    print(np.array(counters) / sum(counters))


if __name__ == "__main__":
    main()
    # s = sum([0.2, 0.2, 0.2, 0.2, 0.2])
    # print(s)
