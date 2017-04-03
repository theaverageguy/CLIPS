
import random
import operator
import numpy as np
import math

orientations = [(1, 0), (0, 1), (-1, 0), (0, -1),(1, 1),(1, -1),(-1, 1),(-1, -1)]


wind = [[(),(),(),(),(),(),()],
        [(),(),(),(),(),(),()],
        [(),(),(),(),(),(),()],
        [(),(),(),(),(),(),()],
        [(),(),(),(),(),(),()],
        [(),(),(),(),(),(),()],
        [(),(),(),(),(),(),()]]


        


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]

def turn_right(heading):
    return turn_heading(heading, -1)


def turn_left(heading):
    return turn_heading(heading, +1)

def isnumber(x):
    """Is x a number?"""
    return hasattr(x, '__int__')

def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))

def print_table(table, header=None, sep='   ', numfmt='%g'):
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]

    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row]
             for row in table]

    sizes = list(
            map(lambda seq: max(map(len, seq)),
                list(zip(*[map(str, row) for row in table]))))

    for row in table:
        print(sep.join(getattr(
            str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))

argmax = max

class MDP:

    def __init__(self, init, actlist, terminals, gamma=.9):
        self.init = init
        self.actlist = actlist
        self.terminals = terminals
        if not (0 <= gamma < 1):
            raise ValueError("An MDP must have 0 <= gamma < 1")
        self.gamma = gamma
        self.states = set()
        self.reward = {}

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(self, state, action):
        raise NotImplementedError

    def actions(self, state):
        if state in self.terminals:
            return [None]
        else:
            return self.actlist


class GridMDP(MDP):

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse()  # because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def P(self,v,w,sigma):
        return np.exp((-0.5)*(math.pow(((v-w)/sigma),2)))




    def integrate(self,w,sigma,l,u):
        return (0.39904*sigma)*[(quad(P,l,u,args=(w,sigma)))][0][0]

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""
        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {
            (1, 0): 'E', (0, 1): 'N', (-1, 0): 'W', (0, -1): 'S',(1, 1): 'NE', (-1, 1): 'NW', (-1, -1): 'SW', (1, -1): 'SE', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})

sequential_decision_environment = GridMDP([[-0.04, -0.04, -0.04,-0.04, -0.04, -0.04, +1],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],
                                           [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(6, 6)])


def value_iteration(mdp, epsilon=0.001):
    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return U


def best_policy(mdp, U):
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
    return pi


def expected_utility(a, s, U, mdp):
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])

def policy_iteration(mdp):
    U = {s: 0 for s in mdp.states}
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi


def policy_evaluation(pi, U, mdp, k=20):
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum([p * U[s1] for (p, s1) in T(s, pi[s])])
    return U


pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .01))
for p in  pi:
    print p,pi[p]


print

sequential_decision_environment.to_arrows(pi)
#[['>', '>', '>', '.'], ['^', None, '^', '.'], ['^', '>', '^', '<']]

print_table(sequential_decision_environment.to_arrows(pi))
print

print_table(sequential_decision_environment.to_arrows(policy_iteration(sequential_decision_environment)))


