import numpy as np
import pandas as pd
import lal
import lalsimulation as lalsim

df = pd.read_csv('./GWTC3_confident.csv', header=0, delimiter=',')

# get mass ratio and chirp mass in detector frame
df['mass_ratio'] = 1. / (df['mass_2_source'] / df['mass_1_source'])
df['chirp_mass'] = df['chirp_mass_source'] * (1. + df['redshift'])
df['mass_1'] = df['mass_1_source'] * (1. + df['redshift'])
df['mass_2'] = df['mass_2_source'] * (1. + df['redshift'])

# calculate the chirp time
chirp_time = []
for index, row in df.iterrows():
    chirp_time.append(
        lalsim.SimInspiralChirpTimeBound(
            20.,
            row['mass_1'] * lal.MSUN_SI,
            row['mass_2'] * lal.MSUN_SI,
            0., 0.
        )
    )

df['chirp_time'] = np.array(chirp_time)

cols_of_interest = [
    'commonName', 'GPS', 'chirp_mass', 'mass_ratio', 'chirp_time',
    'luminosity_distance', 'network_matched_filter_snr'
]

pd.set_option('display.float_format', '{:.10f}'.format)

print(df[cols_of_interest].sort_values(by=['network_matched_filter_snr'], ascending=False))
