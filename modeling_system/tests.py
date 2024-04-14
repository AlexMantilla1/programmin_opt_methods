import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("data.csv")

# Extract each column as a vector
input_signal = df["input"].tolist()
output_signal = df["output"].tolist()

# check
n_input = list(range(len(input_signal)))
n_output = list(range(len(output_signal)))
plt.figure
plt.plot(n_input, input_signal, "--s")
a = plt.plot(n_output, output_signal, "--o")
plt.grid(True)
plt.axis([-1, 40.0, 3.0, 7.0])
plt.legend(["input", "output"])
plt.show()

# Please refer to https://matplotlib.org/stable/tutorials/artists.html
