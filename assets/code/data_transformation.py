def getTaskFromRun(run: int) -> str:
  if run == 1: return tasks["T0A"]
  if run == 2: return tasks["T0B"]
  try: return tasks[f"T{(run-2)%4}"]
  except: return tasks["na"]

def transformRunData(n_subject, n_run):
  # Get data and standarize
  f_name = mne.datasets.eegbci.load_data(n_subject, n_run, path=MNE_PATH)[0]
  raw = mne.io.read_raw_edf(f_name, preload=True).load_data()
  mne.datasets.eegbci.standardize(raw)

  # Rename and get events
  raw.annotations.rename(getAnnotationMapping(n_run))
  events_from_annot, event_dict = mne.events_from_annotations(raw)

  # Get event durations
  # Set second (end) value on annotations to first (start) value of next item -1
  for i in range(len(events_from_annot)):
    if i+1<len(events_from_annot):
      events_from_annot[i][1] = events_from_annot[i+1][0]-1
    else:
      events_from_annot[i, 1] = len(raw)

  # Get data in np arr
  time_data = raw.get_data().transpose()

  def getPaddedRange(a, b):
    start = a
    end = b if b-a<720 else a+720
    ev = time_data[start:end]
    res = np.pad(ev, ((0, 720-len(ev)), (0,0)), 'wrap')
    return res

  # Save all events
  for ev in event_dict:
    events = mne.event.pick_events(events_from_annot, include=[event_dict[ev]])
    data = np.array([getPaddedRange(e[0],e[1])[:] for e in events])
    path = f'data/S{n_subject:03d}/R{n_run:02d}'
    os.makedirs(path, exist_ok=True)
    np.save(f'{path}/{ev}', data)