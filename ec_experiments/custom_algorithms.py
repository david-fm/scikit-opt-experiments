from __future__ import annotations

import numpy as np
from sko.GA import GA, RCGA, GA_TSP
from sko.operators import selection


class TurnSizeConfigurableMixin:
    def __init__(self, *args, window_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size

    def selection(self):
        return selection.selection_tournament_faster(self, tourn_size=self.window_size)


class GA_modified(TurnSizeConfigurableMixin, GA):
    pass


class RCGA_modified(TurnSizeConfigurableMixin, RCGA):
    pass


class GA_TSP_modified(TurnSizeConfigurableMixin, GA_TSP):
    pass


class GA_Binary(GA_modified):
    def bin2rv(self, bin_code):
        _, len_bin_code = bin_code.shape
        mask = np.logspace(start=1, stop=len_bin_code, base=0.5, num=len_bin_code)
        return (bin_code * mask).sum(axis=1) / mask.sum()

    def chrom2x(self, Chrom):
        cumsum_len_segment = self.Lind.cumsum()
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        for i, _ in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = Chrom[:, : cumsum_len_segment[0]]
            else:
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1] : cumsum_len_segment[i]]

            X[:, i] = self.bin2rv(Chrom_temp)

        if self.int_mode:
            X = self.lb + (self.ub_extend - self.lb) * X
            X = np.where(X > self.ub, self.ub, X)
        else:
            X = self.lb + (self.ub - self.lb) * X
        return X
