from baselines.common import plot_util as pu
LOG_DIRS = 'reacher/'
results = pu.load_results(LOG_DIRS)
fig = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False)