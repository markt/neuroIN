from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from mne import events_from_annotations

class Trials():
	'''
	Custom class to store raw data from a trial
	'''

	def __init__(self, fname):
		try:
			raw = read_raw_edf(fname, preload=True)

		except(OSError, FileNotFoundError):
		    print(f'Unable to find {fname}')

		except Exception as error:
		    print(f'Error: {error}')

		# replace channel names
		new_names = {ch: ch.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp') for ch in raw.ch_names}
		raw.rename_channels(new_names)

		# standardize montage
		montage = make_standard_montage('standard_1005')
		raw.set_montage(montage)



		events, blank = events_from_annotations(raws)

		data, times = raw[:, :]
		self.data, self.times = data, times

		trials = []
		labels = []

		mapping = {1:'T0', 2:'T1', 3:'T2'}

		for i in range(len(events) - 1):
		    start, end, label = events[i][0], events[i+1][0], mapping[events[i][2]]
		    trials.append(data[:,start:end])
		    labels.append(label)
		    
		start, end, label = events[-1][0], data.shape[1], mapping[events[-1][2]]
		trials.append(data[:,start:end])
		labels.append(label)

		self.trials = trials
		self.labels = labels