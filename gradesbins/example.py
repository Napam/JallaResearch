import numpy as np
scores = [95, 100, 80, 42, 39, 96, 80, 69]
bins = [64, 66, 69, 72, 76, 79, 82, 86, 89, 92, 96, 100]
gpa_scale = np.array([0.0, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0, 4.0])
print(repr(gpa_scale[np.searchsorted(bins, scores, side='left')]))
