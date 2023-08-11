import sciris as sc
import hpvsim as hpv
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib as mpl
# local imports
import network_pars as pn
from analyzers import new_pairs_snap


def run_network_develop(start, end, pop, location):

    #labels = ['Clustered network', 'Status quo']
    labels = []
    snap = hpv.snapshot(
        timepoints=['1990', '2000', '2010', '2020'],
    )
    snaps = []
    new_pairs = new_pairs_snap(start_year = 2015)
    df_new_pairs = pd.DataFrame(columns = ['f', 'm', 'acts', 'dur', 'start', 'end', 'age_f', 'age_m', 'year', 'rtype', 'sim'])
    pars = dict(
        n_agents=pop,
        start=start,
        end=end,
        location=location,
        debut = pn.debut[location],
        mixing = pn.mixing[location],
        layer_probs = pn.layer_probs[location],
        dur_pship = pn.dur_pship[location],
        partners = pn.partners[location],
        ms_agent_ratio=100,
        analyzers=[snap, new_pairs]
    )
    i = 0
    sim = hpv.Sim(pars=pars)
    sim.run()
    # Plot age mixing
    snaps.append(sim.get_analyzer([0]))
    new_pairs_snaps = sim.get_analyzer([1]).new_pairs
    new_pairs_snaps['sim'] = i
    df_new_pairs = pd.concat([df_new_pairs, new_pairs_snaps])
    ## Network diagnostics
    plot_mixing(sim, df_new_pairs)

    fig0, axes = pl.subplots(2, 1)
    axes[0].plot(sim.results['year'], sim.results['infections'])
    axes[1].plot(sim.results['year'], sim.results['cancers'])


    axes[0].legend()
    axes[0].set_ylabel('Infections')
    axes[1].set_ylabel('Cancers')
    fig0.show()

    fig, axes = pl.subplots(nrows=2, ncols=3, figsize=(14, 10))
    for i, isnap in enumerate(snaps):
        people2020 = isnap.snapshots[3]
        font_size = 15
        font_family = 'Libertinus Sans'
        pl.rcParams['font.size'] = font_size
        pl.rcParams['font.family'] = font_family

        # ax = axes.flatten()
        people = people2020
        rships_f = np.zeros((3, len(people.age_bin_edges)))
        rships_m = np.zeros((3, len(people.age_bin_edges)))
        for lk, lkey in enumerate(['m', 'c', 'o']):
            active_ages = people.age#[(people.n_rships[lk,:] >= 1)]
            n_rships = people.n_rships#[:,(people.n_rships[lk,:] >= 1)]
            age_bins = np.digitize(active_ages, bins=people.age_bin_edges) - 1

            for ab in np.unique(age_bins):
                inds_f = (age_bins==ab) & people.is_female
                inds_m = (age_bins==ab) & people.is_male
                rships_f[lk,ab] = n_rships[lk,inds_f].sum()/len(hpv.true(inds_f))
                rships_m[lk, ab] = n_rships[lk, inds_m].sum() / len(hpv.true(inds_m))
            ax = axes[0, lk]
            yy_f = rships_f[lk,:]
            yy_m = rships_m[lk,:]
            ax.bar(people.age_bin_edges-1, yy_f, width=1.5, label='Female')
            ax.bar(people.age_bin_edges+1, yy_m, width=1.5, label='Male')
            ax.set_xlabel(f'Age')
            ax.set_title(f'Average number of relationships, {lkey}')
            #axes[0].set_ylabel(labels[i])
            axes[0,2].legend()
        fig.tight_layout()
        #fig.show()

    for i, isnap in enumerate(snaps):
        people2020 = isnap.snapshots[3]
        font_size = 15
        font_family = 'Libertinus Sans'
        pl.rcParams['font.size'] = font_size
        pl.rcParams['font.family'] = font_family

        # ax = axes.flatten()
        people = people2020

        types = ['marital', 'casual', 'one-off']
        xx = people.lag_bins[1:15] * sim['dt']
        for cn, lkey in enumerate(['m', 'c', 'o']):
            ax = axes[1, cn]
            yy = people.rship_lags[lkey][:14] / sum(people.rship_lags[lkey])
            ax.bar(xx, yy, width=0.2)
            ax.set_xlabel(f'Time between {types[cn]} relationships')
        #axes[i,0].set_ylabel(labels[i])

    fig.tight_layout()
    fig.show()

    print('done')

def plot_mixing(sim, df_new_pairs, by_year = 3):
    for runind in df_new_pairs.sim.unique():
        for i, rtype in enumerate(['m','c','o']):
            df = df_new_pairs[(df_new_pairs['sim'] == runind) & (df_new_pairs['rtype'] == rtype)]
            n_time = len(df_new_pairs.year.unique())
            check_square = n_time % np.sqrt(n_time)
            non_square = 1
            if check_square == 0:
                nr = int(np.sqrt(n_time))
                nc = int(np.sqrt(n_time))
            else:
                nr = int(np.sqrt(n_time)) + non_square
                nc = int(np.sqrt(n_time))
                if nr * nc < n_time:
                    nc += 1
            fig, ax = pl.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True, figsize=(15, 12))
            for j, year in enumerate(df_new_pairs.year.unique()):
                df_year = df[df['year']==year]
                fc = df_year.age_f  # Get the age of female contacts in marital partnership
                mc = df_year.age_m  # Get the age of male contacts in marital partnership
                h = ax[j//nc, j%nc].hist2d(fc, mc, bins=np.linspace(0, 75, 16), density=False, norm=mpl.colors.LogNorm())
                ax[j//nc, j%nc].set_title(year)

            fig.colorbar(h[3], ax=ax)
            mixing = sim['mixing'][rtype]
            age_bins = mixing[:,0]
            mixing = mixing[:,1:]
            mixing_norm_col = mixing / mixing.max(axis=0)
            mixing_norm_col[np.isnan(mixing_norm_col)] = 0
            #X, Y = np.meshgrid(age_bins, age_bins)
            #h = ax[nr-1, nc-1].pcolormesh(X, Y, mixing_norm_col, norm=mpl.colors.LogNorm())
            #ax[nr-1, nc-1].set_title('Input')

            fig.text(0.5, 0.04, 'Age of female partner', ha='center', fontsize=24)
            fig.text(0.04, 0.5, 'Age of male partner', va='center', rotation='vertical', fontsize=24)

            fig.suptitle(rtype, fontsize=24)
            fig.tight_layout(h_pad=0.5)
            fig.subplots_adjust(top=0.9, left=0.1, bottom=0.1, right=0.75)
            fig.show()

# %% Run as a script
if __name__ == '__main__':
    # Start timing and optionally enable interactive plotting
    T = sc.tic()
    #geos = [25, 25, 25, 5]
    #geo_mix = [[0], [0.5, 0.01], np.repeat(1,24), np.repeat(1,4)]
    start = 1970
    end = 2020
    pop = 10e5
    location = 'india'
    #sim = run_experiment(geos, geo_mix, start, end, pop)
    #
    sim1 = run_network_develop(start, end, pop, location)
    #sim1 = run_init(geos,geo_mix,start,end,pop)

    sc.toc(T)
    print('Done.')