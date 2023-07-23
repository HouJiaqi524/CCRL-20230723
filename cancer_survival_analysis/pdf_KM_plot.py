import pandas as pd
import pyarrow.parquet as pq
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt

survival_label = pq.read_table('out - 21ensembles/3_cluster_results/trial_7/survival_label.parquet').to_pandas()
label_grouped = survival_label.groupby('ensemble')
fig, ax = plt.subplots(figsize=(20, 16))
a = {}
for label, survival in label_grouped:
    a[str(label)] = survival
    kmf1 = KaplanMeierFitter().fit(survival['time'], survival['event'], label=label)
    ax = kmf1.plot_survival_function(ax=ax)
plt.xlabel('time', fontsize=20)
plt.ylabel('S(t)', fontsize=20)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(r'out - 21ensembles/4_KM_curve/trial_7/km_plot.pdf')
plt.close()
p_value = logrank_test(a['0']['time'], a['1']['time'], a['0']['event'], a['1']['event']).p_value
print(p_value)