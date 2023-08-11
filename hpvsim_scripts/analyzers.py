'''
Define custom analyzers for use case
'''

import numpy as np
import sciris as sc
import hpvsim as hpv
import pandas as pd
import pylab as pl


#%% Define analyzer for computing DALYs
class dalys(hpv.Analyzer):

    def __init__(self, max_age=84, cancer=None, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age
        self.cancer = cancer if cancer else dict(dur=1, wt=0.16325) # From GBD 2017, calculated as 1 yr @ 0.288 (primary), 4 yrs @ 0.049 (maintenance), 0.5 yrs @ 0.451 (late), 0.5 yrs @ 0.54 (tertiary), scaled up to 12 years
        return

    def initialize(self, sim):
        super().initialize(sim)
        return

    def apply(self, sim):
        pass

    def finalize(self, sim):
        scale = sim['pop_scale']

        # Years of life lost
        dead = sim.people.dead_cancer
        years_left = np.maximum(0, self.max_age - sim.people.age)
        self.yll = (years_left*dead).sum()*scale
        self.deaths = dead.sum()*scale

        # Years lived with disability
        cancer = sc.objdict(self.cancer)
        n_cancer = (sim.people.cancerous).sum()*scale
        self.n_cancer = n_cancer
        self.yld = n_cancer*cancer.dur*cancer.wt
        self.dalys = self.yll + self.yld
        return


class new_pairs_snap(hpv.Analyzer):
    def __init__(self, start_year=None, by_year=3, **kwargs):
        super().__init__(**kwargs)
        self.new_pairs = pd.DataFrame(columns = ['f', 'm', 'acts', 'dur', 'start', 'end', 'age_f', 'age_m', 'year', 'rtype'])
        self.start_year = start_year
        self.yearvec = None
        self.by_year = by_year

    def initialize(self, sim):
        super().initialize()
        self.yearvec = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            tind = sim.yearvec[sim.t] - sim['start']
            for rtype in ['m','c','o']:
                new_rship_inds = (sim.people.contacts[rtype]['start'] == tind).nonzero()[0]
                if len(new_rship_inds):
                    contacts = pd.DataFrame.from_dict(sim.people.contacts[rtype].get_inds(new_rship_inds))
                    #contacts = pd.DataFrame.from_dict(sim.people.contacts[rtype])
                    contacts['year'] = int(sim.yearvec[sim.t])
                    contacts['rtype'] = rtype
                    self.new_pairs = pd.concat([self.new_pairs, contacts])
        return


    def plot(self, do_save=False, filename=None, ag=False):
        n_time = len(self.new_pairs[0])
        check_square = n_time % np.sqrt(n_time)
        non_square = 1
        if check_square == 0:
            nrows = int(np.sqrt(n_time))
            ncols = int(np.sqrt(n_time))
        else:
            nrows = int(np.sqrt(n_time)) + non_square
            ncols = int(np.sqrt(n_time))


        fig, ax = pl.subplots(nrows, ncols, figsize=(15, 8))
        yi = sc.findinds(self.yearvec, from_when)[0]
        for rn, rtype in enumerate(['m', 'c', 'o']):
            ax[0, rn].plot(self.yearvec[yi:], self.n_edges[rtype][yi:])
            ax[0, rn].set_title(f'Edges - {rtype}')
            ax[1, rn].plot(self.yearvec[yi:], self.n_edges_norm[rtype][yi:])
            ax[1, rn].set_title(f'Normalized edges - {rtype}')
        pl.tight_layout()
        if do_save:
            fn = 'networks' or filename
            fig.savefig(f'{filename}.png')
        else:
            pl.show()
