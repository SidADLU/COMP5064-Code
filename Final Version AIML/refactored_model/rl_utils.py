"""
rl_utils.py -- two small reinforcement-learning helpers kept for a possible later RL stage on the exemplar model.

RollingMultiset     bounded memory of recent outputs with O(1) add / count (for a self-diversity penalty)
PerContextBaseline  running per-context EMA baseline for advantages; the context key can be any hashable (for the exemplar model: the seed SMILES)

The rest of the old reward code (context-relevance index, reward calculator, ID-based context key) belongs to the retired ID-based pipeline and is archived in legacy/rl_fixes.py.
"""
from collections import Counter, deque
from typing import Deque, Dict, Tuple


# A bounded multiset of the most recent items.  Once more than maxlen items have been added the oldest is evicted, so a repeat penalty based on count() fades as old output ages out of the window.
class RollingMultiset:

    # maxlen is the window size; _order keeps arrival order, _counts the number of copies of each item currently in the window.
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._order: Deque[str] = deque()
        self._counts: Counter = Counter()

    # Adds one item and, if the window is now too long, removes the oldest one (dropping its count entry when it reaches zero).
    def add(self, item: str) -> None:
        self._order.append(item)
        self._counts[item] += 1
        if len(self._order) > self.maxlen:
            oldest = self._order.popleft()
            self._counts[oldest] -= 1
            if self._counts[oldest] <= 0:
                del self._counts[oldest]

    # How many copies of the item are currently in the window (0 if none).
    def count(self, item: str) -> int:
        return self._counts.get(item, 0)


# Running reward baseline per context, as an exponential moving average, for computing advantages in REINFORCE.  A context never seen before gets the global running mean (a neutral start rather than 0).
# Call update() for every sample AFTER computing the batch's advantages, never before, or a sample would be compared with a baseline that already contains its own reward.
class PerContextBaseline:

    # ema_decay is the weight of the old value in each update; context_baseline maps a context key to its running mean, global_mean is the same over all contexts.
    def __init__(self, ema_decay: float = 0.9):
        self.ema_decay = ema_decay
        self.context_baseline: Dict[Tuple, float] = {}
        self.global_mean = 0.0
        self._global_initialized = False

    # Current baseline for a context: its own running mean, or the global mean if the context is new.
    def get(self, context_key: Tuple) -> float:
        return self.context_baseline.get(context_key, self.global_mean)

    # Folds one reward into the context's running mean (the first reward of a context starts it) and into the global mean.
    def update(self, context_key: Tuple, reward: float) -> None:
        if context_key in self.context_baseline:
            old = self.context_baseline[context_key]
            self.context_baseline[context_key] = self.ema_decay * old + (1 - self.ema_decay) * reward
        else:
            self.context_baseline[context_key] = reward

        if not self._global_initialized:
            self.global_mean = reward
            self._global_initialized = True
        else:
            self.global_mean = self.ema_decay * self.global_mean + (1 - self.ema_decay) * reward
