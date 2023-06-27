import pandas as pd
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

# Read data from CSV
data = pd.read_csv('agricultural_data.csv')

# Separate independent and dependent variables
X = data[['Nutrient', 'Irrigation', 'Pesticide', 'Sunlight']]
Y = data[['Yield', 'Height', 'Weight']]

# Perform Canonical regression
cca = CCA(n_components=2)
X_c, Y_c = cca.fit_transform(X, Y)

# Visualize the results
plt.scatter(X_c[:, 0], X_c[:, 1], c='r', label='Independent Variables')
plt.scatter(Y_c[:, 0], Y_c[:, 1], c='b', label='Dependent Variables')
plt.xlabel('Canonical Variable 1')
plt.ylabel('Canonical Variable 2')
plt.legend()
plt.show()
