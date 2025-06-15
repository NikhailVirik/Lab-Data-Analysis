###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import read_data_results3 as rd

#Step 1 get the data and the x position
#file='%ii.txt'%(sys.argv[1]) #this is the data
results = np.loadtxt(r'C:\Users\Nikhail\Downloads\W_LED_6.txt')

y1 = np.array(results[:,0])
y2 = np.array(results[:,1])

x=np.array(results[:,5])

plt.figure("Detector 1")
plt.plot(x,y1,'o-')
plt.xlabel("Position $\mu$steps]")
plt.ylabel("Signal 1")
#plt.savefig("figures/quick_plot_detector_1.png")
plt.title('White LED Interferogram')

# plt.figure("Detector 2")
# plt.plot(x,y2,'o-')
# plt.xlabel("Position $\mu$steps]")
# plt.ylabel("Signal 2")
# plt.savefig("figures/quick_plot_detector_2.png")

plt.show()
