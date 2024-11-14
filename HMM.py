

import random
import argparse
import codecs
import os
import numpy
from sympy.physics.units import current


# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""


        self.transitions = {}
        self.emissions = {}

        with open(f"{basename}.trans") as f:
            for line in f:

                # check empty line
                if not line.strip():
                    continue

                parts = line.strip().split()

                if parts[0] == '#':
                    if '#' not in self.transitions:
                        self.transitions['#'] = {}
                    self.transitions['#'][parts[1]] = float(parts[2])
                else:
                    from_state = parts[0]
                    to_state = parts[1]
                    prob = parts[2]

                    # nested dictionary
                    if from_state not in self.transitions:
                        self.transitions[from_state] = {}
                    self.transitions[from_state][to_state] = prob

        with open(f"{basename}.emit") as f:
            for line in f:
                # check empty line
                if not line.strip():
                    continue
                state, emission, prob = line.strip().split()
                if state not in self.emissions:
                    self.emissions[state] = {}
                self.emissions[state][emission] = prob



    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""

        state_sequence = []
        emission_sequence = []

        # get the initial state
        initial_prob = self.transitions['#']
        states = list(initial_prob.keys())
        probs = [float(initial_prob[state]) for state in states]
        current_state = numpy.random.choice(states, p=probs)

        # add the initial state to seqquence
        state_sequence.append(current_state)

        # generate emission for initial state

        emit_options = list(self.emissions[current_state].keys())
        emit_probs = [float(self.emissions[current_state][e]) for e in emit_options]
        emission = numpy.random.choice(emit_options, p=emit_probs)
        emission_sequence.append(emission)

        # remaining n-1 states transitions and emissions

        for _ in range(n-1):
            # trans
            next_states = list(self.transitions[current_state].keys())
            next_probs = [float(self.transitions[current_state][s]) for s in next_states]
            current_state = numpy.random.choice(next_states, p=next_probs)
            state_sequence.append(current_state)
            # emit
            emit_options = list(self.emissions[current_state].keys())
            emit_probs = [float(self.emissions[current_state][e]) for e in emit_options]
            emission = numpy.random.choice(emit_options, p=emit_probs)
            emission_sequence.append(emission)

        return Sequence(state_sequence, emission_sequence)



    def forward(self, sequence):

        states = [state for state in self.transitions.keys() if state != '#']

        T = len(sequence.outputseq)  # Length of observation sequence
        M = {t: {s: 0.0 for s in states} for t in range(T)}

        first_obs = sequence.outputseq[0]
        for s in states:
            M[0][s] = float(self.transitions['#'][s]) * float(self.emissions[s][first_obs])

        for t in range(1, T):
            obs = sequence.outputseq[t]
            for s in states:
                sum_prob = 0
                for s2 in states:
                    sum_prob += M[t - 1][s2] * float(self.transitions[s2][s])
                M[t][s] = sum_prob * float(self.emissions[s][obs])

        # find most likely final state
        final_timestep = T - 1
        max_prob = -1
        most_likely_state = None

        # go through each state in final timestep
        for state in states:
            prob = M[final_timestep][state]
            if prob > max_prob:
                max_prob = prob
                most_likely_state = state

        return most_likely_state




    def viterbi(self, sequence):
        states = [state for state in self.transitions.keys() if state != '#']
        T = len(sequence.outputseq)

        # matrices for probabilities and backpointers
        M = {t: {s: 0.0 for s in states} for t in range(T)}
        Backpointers = {t: {s: None for s in states} for t in range(T)}

        first_obs = sequence.outputseq[0]
        for s in states:
            M[0][s] = float(self.transitions['#'][s]) * float(self.emissions[s][first_obs])
            Backpointers[0][s] = '#'


        for t in range(1, T):
            obs = sequence.outputseq[t]
            for s in states:
                # find most likely previous state
                max_prob = -1
                best_prev_state = None

                for prev_s in states:
                    prob = M[t - 1][prev_s] * float(self.transitions[prev_s][s]) * float(self.emissions[s][obs])
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = prev_s

                M[t][s] = max_prob
                Backpointers[t][s] = best_prev_state

        # find most likely final state
        final_timestep = T - 1
        max_prob = -1
        current_state = None

        # find most probable final state
        for state in states:
            prob = M[final_timestep][state]
            if prob > max_prob:
                max_prob = prob
                current_state = state

        # backtrack for sequence
        state_sequence = []
        for t in range(final_timestep, -1, -1):
            state_sequence.insert(0, current_state)
            current_state = Backpointers[t][current_state]

        return state_sequence






def main():
    parser = argparse.ArgumentParser(description='Generate sequences using HMM')
    parser.add_argument('model', type=str, choices=['cat', 'partofspeech'])
    parser.add_argument('--generate', type=int, required=True)

    args = parser.parse_args()

    #load HMM model
    model = HMM()
    model.load(args.model)

    sequence = model.generate(args.generate)

    print(sequence)


if __name__ == "__main__":
    main()