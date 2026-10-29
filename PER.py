"""Prioritized replay with partial-buffer sampling and recurrent context."""
import numpy as np


class SumTree:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0
        self.insertion_ids = np.full(capacity, -1, dtype=np.int64)
        self.counter = 0

    def add(self, priority, data):
        slot = self.data_pointer
        self.data[slot] = np.asarray(data, dtype=np.float32).copy()
        self.insertion_ids[slot] = self.counter
        self.update(slot + self.capacity - 1, priority)
        self.counter += 1
        self.size = min(self.size + 1, self.capacity)
        self.data_pointer = (slot + 1) % self.capacity

    def update(self, index, priority):
        priority = float(priority)
        if not np.isfinite(priority) or priority <= 0:
            raise ValueError("priority must be finite and positive")
        change = priority - self.tree[index]
        self.tree[index] = priority
        while index:
            index = (index - 1) // 2
            self.tree[index] += change

    def get_leaf(self, value):
        value = np.clip(value, 0, np.nextafter(self.total_p, 0))
        index = 0
        while 2 * index + 1 < len(self.tree):
            left = 2 * index + 1
            if value < self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = left + 1
        return index, self.tree[index], self.data[index - self.capacity + 1]

    @property
    def total_p(self):
        return self.tree[0]


class PrioritizedExperienceReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, seed=None):
        self.tree = SumTree(capacity)
        self.alpha, self.beta, self.beta_increment = alpha, beta, beta_increment
        self.epsilon = 0.01
        self.max_priority = 1.0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.tree.size

    def store(self, transition):
        self.tree.add(self.max_priority, transition)

    def sample(self, count):
        if len(self) < count:
            raise ValueError("Not enough stored transitions for a batch")
        self.beta = min(1.0, self.beta + self.beta_increment)
        indices, probabilities, rows = [], [], []
        for i in range(count):
            value = self.rng.uniform(i, i + 1) * self.tree.total_p / count
            index, priority, row = self.tree.get_leaf(value)
            indices.append(index)
            probabilities.append(priority / self.tree.total_p)
            rows.append(row)
        leaves = self.tree.tree[-self.tree.capacity:]
        min_probability = leaves[leaves > 0].min() / self.tree.total_p
        weights = (np.asarray(probabilities) / min_probability) ** (-self.beta)
        return np.array(indices), np.stack(rows), weights.astype(np.float32).reshape(-1, 1)

    def batch_update(self, indices, abs_errors):
        if hasattr(abs_errors, "detach"):
            abs_errors = abs_errors.detach().cpu().numpy()
        errors = np.asarray(abs_errors).reshape(-1)
        if len(errors) != len(indices) or not np.isfinite(errors).all():
            raise ValueError("Expected one finite TD error per sampled transition")
        priorities = (np.abs(errors) + self.epsilon) ** self.alpha
        for index, priority in zip(indices, priorities):
            self.tree.update(int(index), priority)
        self.max_priority = max(self.max_priority, float(priorities.max()))

    def state_sequences(self, indices, state_dim, sequence_length):
        """Rebuild causal context, never crossing resets or overwritten ring entries.

        Rows end with [terminated, episode_end, next_state...]. Repeat the first
        available state at the left boundary, matching the acting policy.
        """
        sequences = []
        for index in indices:
            slot = int(index) - self.tree.capacity + 1
            states = [self.tree.data[slot][:state_dim]]
            current_id = self.tree.insertion_ids[slot]
            for _ in range(sequence_length - 1):
                previous = (slot - 1) % self.tree.capacity
                if self.tree.insertion_ids[previous] < 0 or self.tree.insertion_ids[previous] != current_id - 1:
                    break
                row = self.tree.data[previous]
                if bool(row[-state_dim - 1]):  # episode_end includes time limits
                    break
                states.append(row[:state_dim])
                slot, current_id = previous, current_id - 1
            states.reverse()
            sequences.append(np.stack([states[0]] * (sequence_length - len(states)) + states))
        return np.stack(sequences)
