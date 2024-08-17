import numpy as np
import torch
from matplotlib import pyplot as plt


def is_met(scale, l2_err, threshold):
    return (l2_err / scale) < threshold


def key_is_met(metric_cache, config, ep_len, target_key, env_id, threshold):
    # metric_cache[target_key][env_id] / ep_len
    scale = 1
    l2_err = 0
    return is_met(scale, l2_err, threshold)


class Curriculum:
    def __init__(self, seed, **key_ranges):

        self.rng = np.random.RandomState(seed)
        self.num_commands = len(key_ranges.values())

        values = []
        indices = []
        self.bin_sizes = []  # len: num_commands
        self.num_bins = []   # len: num_commands
        self.lows = []
        self.highs = []
        for v_range in key_ranges.values():

            min_v, max_v, n_bin = v_range
            bin_size = (max_v - min_v) / n_bin

            self.bin_sizes.append(bin_size)
            self.num_bins.append(n_bin)
            self.lows.append(min_v)
            self.highs.append(max_v)

            values.append(np.linspace(min_v + bin_size / 2, max_v - bin_size / 2, n_bin))
            indices.append(np.linspace(0, n_bin - 1, n_bin))

        self.bin_sizes = np.array(self.bin_sizes)
        self.lows = np.array(self.lows)
        self.highs = np.array(self.highs)

        self._raw_grid = np.stack(np.meshgrid(*values, indexing='ij'))   # shape: (num_commands, *bin_sizes)
        self._idx_grid = np.stack(np.meshgrid(*indices, indexing='ij'))  # shape: (num_commands, *bin_sizes)
        self.keys = [*key_ranges.keys()]
        self.grid = self._raw_grid.reshape([self.num_commands, -1])      # shape: (num_commands, num_bins_total)
        self.idx_grid = self._idx_grid.reshape([self.num_commands, -1])  # shape: (num_commands, num_bins_total)

        self.num_bins_total = self.grid.shape[1]

        self.weights = np.zeros(self.num_bins_total)
        self.indices = np.arange(self.num_bins_total)

    def __len__(self):
        return self.num_bins_total

    def __getitem__(self, *keys):
        pass

    def set_to(self, low, high, value=1.0):
        """ Set the weights of commands in grid that meet low <= command <= high to value """

        inds = np.logical_and(
            self.grid >= low[:, None],
            self.grid <= high[:, None]
        ).all(axis=0)

        assert len(inds) != 0, "You are intializing your distribution with an empty domain!"

        self.weights[inds] = value

    def update(self, **kwargs):
        # bump the envelop if
        pass

    def sample_bins(self, batch_size, low=None, high=None):
        """sample bins according to weights"""
        if low is not None and high is not None: # if bounds given
            valid_inds = np.logical_and(
                self.grid >= low[:, None],
                self.grid <= high[:, None]
            ).all(axis=0)
            temp_weights = np.zeros_like(self.weights)
            temp_weights[valid_inds] = self.weights[valid_inds]
            inds = self.rng.choice(self.indices, batch_size, p=temp_weights / temp_weights.sum())
        else: # if no bounds given
            inds = self.rng.choice(self.indices, batch_size, p=self.weights / self.weights.sum())

        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        low, high = centroids - self.bin_sizes / 2, centroids + self.bin_sizes / 2
        return self.rng.uniform(low, high)

    def sample(self, batch_size, low=None, high=None):
        cgf_centroid, inds = self.sample_bins(batch_size, low=low, high=high)
        return self.sample_uniform_from_cell(cgf_centroid), inds


class SumCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

        self.success = np.zeros(len(self))
        self.trials = np.zeros(len(self))

    def update(self, bin_inds, l1_error, threshold):
        is_success = l1_error < threshold
        self.success[bin_inds[is_success]] += 1
        self.trials[bin_inds] += 1

    def success_rates(self, *keys):
        s_rate = self.success / (self.trials + 1e-6)
        s_rate = s_rate.reshape(self.num_bins)
        marginals = tuple(i for i, key in enumerate(self.keys) if key not in keys)
        if marginals:
            return s_rate.mean(axis=marginals)
        return s_rate


class RewardThresholdCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

        self.episode_reward_lin = np.zeros(len(self))
        self.episode_reward_ang = np.zeros(len(self))
        self.episode_lin_vel_raw = np.zeros(len(self))
        self.episode_ang_vel_raw = np.zeros(len(self))
        self.episode_duration = np.zeros(len(self))

    def get_local_bins(self, bin_inds, ranges=0.1):
        if isinstance(ranges, float):
            ranges = np.ones(self.grid.shape[0]) * ranges
        bin_inds = bin_inds.reshape(-1)

        adjacent_inds = np.logical_and(
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) >= self.grid[:, bin_inds, None] - ranges.reshape(-1, 1, 1),
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) <= self.grid[:, bin_inds, None] + ranges.reshape(-1, 1, 1)
        ).all(axis=0)

        return adjacent_inds

    def update(self, bin_inds, task_rewards, success_thresholds, local_range=0.5):

        is_success = 1.
        for task_reward, success_threshold in zip(task_rewards, success_thresholds):
            is_success = is_success * (task_reward > success_threshold).cpu()
        if len(success_thresholds) == 0:
            is_success = np.array([False] * len(bin_inds))
        else:
            is_success = np.array(is_success.bool())

        # if len(is_success) > 0 and is_success.any():
        #     print("successes")

        self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
        adjacents = self.get_local_bins(bin_inds[is_success], ranges=local_range)
        for adjacent in adjacents:
            #print(adjacent)
            #print(self.grid[:, adjacent])
            adjacent_inds = np.array(adjacent.nonzero()[0])
            self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)

    def log(self, bin_inds, lin_vel_raw=None, ang_vel_raw=None, episode_duration=None):
        self.episode_lin_vel_raw[bin_inds] = lin_vel_raw.cpu().numpy()
        self.episode_ang_vel_raw[bin_inds] = ang_vel_raw.cpu().numpy()
        self.episode_duration[bin_inds] = episode_duration.cpu().numpy()

if __name__ == '__main__':
    r = RewardThresholdCurriculum(100, x=(-1, 1, 5), y=(-1, 1, 2), z=(-1, 1, 11))

    assert r._raw_grid.shape == (3, 5, 2, 11), "grid shape is wrong: {}".format(r.grid.shape)

    low, high = np.array([-0.6, -0.6, -1.0]), np.array([0.6, 0.6, 1.0])

    r.set_to(low, high, value=1.0)

    adjacents = r.get_local_bins(np.array([10, ]), ranges=0.5)
    for adjacent in adjacents:
        adjacent_inds = np.array(adjacent.nonzero()[0])
        print(adjacent_inds)
        r.update(bin_inds=adjacent_inds, lin_vel_rewards=np.ones_like(adjacent_inds),
                    ang_vel_rewards=np.ones_like(adjacent_inds), lin_vel_threshold=0.0, ang_vel_threshold=0.0,
                    local_range=0.5)

    samples, bins = r.sample(10_000)

    plt.scatter(*samples.T[:2])
    plt.show()
