import numpy as np


class HiddenMarkovChain(object):
    def __init__(self, transition_prob, emission_prob):
        """
        Initialize the MarkovChain instance.

        Parameters
        ----------
        transition_prob: dict
            A dict object representing the transition
            probabilities in Markov Chain.
            Should be of the form:
                {'state1': {'state1': 0.1, 'state2': 0.4},
                 'state2': {...}}
        """
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.states = list(transition_prob.keys())
        self.hiddenStates = list(emission_prob[self.states[0]].keys())

    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time
        instance.

        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return np.random.choice(
            self.states,
            p=[self.transition_prob[current_state][next_state]
               for next_state in self.states]
        )

    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: str
            The state of the current random variable.

        no: int
            The number of future states to generate.
        """
        hidden_states = []
        for i in range(no):

            next_state = self.next_state(current_state)
            next_hidden_state = np.random.choice(self.hiddenStates, p = [self.emission_prob[current_state][cue] for cue in self.hiddenStates])
            hidden_states.append(next_hidden_state)
            current_state = next_state

        return hidden_states

class HiddenMarkovChain2(object):
    def __init__(self, transition_prob, emission_prob):
        """
        Initialize the MarkovChain instance.

        Parameters
        ----------
        transition_prob: dict
            A dict object representing the transition
            probabilities in Markov Chain.
            Should be of the form:
                {'state1': {'state1': 0.1, 'state2': 0.4},
                 'state2': {...}}
        """
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.states = list(transition_prob.keys())
        self.observedStates = list(emission_prob[self.states[0]].keys())

    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time
        instance.

        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return np.random.choice(
            self.states,
            p=[self.transition_prob[current_state][next_state]
               for next_state in self.states]
        )

    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: str
            The state of the current random variable.

        no: int
            The number of future states to generate.
        """
        observed_states = []
        hidden_states = []

        for i in range(no):
            hidden_states.append(current_state)
            next_state = self.next_state(current_state)
            next_observed_state = np.random.choice(self.observedStates, p = [self.emission_prob[current_state][cue] for cue in self.observedStates])
            observed_states.append(next_observed_state)
            current_state = next_state

        return [observed_states, hidden_states]

class MarkovChain(object):
    def __init__(self, transition_prob):
        """
        Initialize the MarkovChain instance.

        Parameters
        ----------
        transition_prob: dict
            A dict object representing the transition
            probabilities in Markov Chain.
            Should be of the form:
                {'state1': {'state1': 0.1, 'state2': 0.4},
                 'state2': {...}}
        """
        self.transition_prob = transition_prob
        self.states = list(transition_prob.keys())

    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time
        instance.

        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return np.random.choice(
            self.states,
            p=[self.transition_prob[current_state][next_state]
               for next_state in self.states]
        )

    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: str
            The state of the current random variable.

        no: int
            The number of future states to generate.
        """
        future_states = [current_state]
        for i in range(no-1):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states

