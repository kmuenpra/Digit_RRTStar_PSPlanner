
# import the library 
import matplotlib.pyplot as plt 
import csv
import pylab
import numpy as np

def read_file(filename):
    # Initialize empty lists for each column
    decision_epoch = []
    avg_std = []
    sum_std = []
    avg_error = []
    sum_error = []

    # Open the CSV file
    with open(filename, newline='') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        
        # Skip the header row
        next(csvreader)
        
        # Iterate over each row in the CSV
        for row in csvreader:
            # Append each value to the corresponding list
            decision_epoch.append(float(row[0]))
            avg_std.append(float(row[1]))
            sum_std.append(float(row[2]))
            avg_error.append(float(row[3]))
            sum_error.append(float(row[4]))

    # # Now you have separate lists for each column of the CSV file
    # print("Decision Epoch:", decision_epoch)
    # print("Avg Std:", avg_std)
    # print("Sum Std:", sum_std)
    # print("Avg Error:", avg_error)
    # print("Sum Error:", sum_error)
    
    return decision_epoch, avg_std, sum_std, avg_error, sum_error


  
decision_epoch_ak, avg_std_ak, sum_std, avg_error, sum_error = read_file("/home/kmuenpra/git/Lantao_Liu/elevation_mapping_with_ak/logs_GP_prediction/eval-AK-data-2024-03-15-18-33-23.csv")
decision_epoch_rbf, avg_std_rbf, sum_std, avg_error, sum_error = read_file("/home/kmuenpra/git/Lantao_Liu/elevation_mapping_with_ak/logs_GP_prediction/eval-RBF-data-2024-03-15-18-54-27.csv")
decision_epoch_nn, avg_std_nn, sum_std, avg_error, sum_error = read_file("/home/kmuenpra/git/Lantao_Liu/elevation_mapping_with_ak/logs_GP_prediction/eval-NN-data-2024-03-15-19-11-34.csv")
  

# avg_std_ak = normalize_value(avg_std_ak)

# avg_std_rbf= normalize_value(avg_std_rbf)

# avg_std_nn = normalize_value(avg_std_nn)
avg_std_nn_shifted = np.array(avg_std_nn) - 0.83
  
fig = pylab.figure()
ax = fig.add_subplot(111)
ax.plot(decision_epoch_ak , avg_std_ak, 'o:r', linestyle='--', linewidth='3',label='AK') 
ax.plot(decision_epoch_rbf , avg_std_rbf, 'o:g', linestyle='--', linewidth='3',label='RBF') 
ax.plot(decision_epoch_nn , avg_std_nn_shifted, 'o:b', linestyle='--', linewidth='3',label='NN') 
ax.set_yticks(np.linspace(0.06, 0.2, 7))
ax.set_yticklabels(["0.06", "0.08", "0.1", "0.12" , "...", "1.02", "1.04"]) 
# ax.legend(('ClassificationAccuracy','One-Error','HammingLoss'),loc='upper right')
ax.set_ylabel('Avg. Standard Deviation of the Environment') 
ax.set_xlabel('Iterations') 
ax.legend()

pylab.show()

 
# Show plot
plt.show()

plt.savefig("Uncertainty_compare", format="svg")