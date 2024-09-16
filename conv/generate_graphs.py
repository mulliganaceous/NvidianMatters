import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_results(filename):
    file = open("outputs/{0:s}.txt".format(filename), "r")

    cpu = []
    kernel1 = []
    transfer_in_1 = []
    transfer_out_1 = []
    speedup_1 = []
    speedup_1_noTrf = []

    kernel2 = []
    transfer_in_2 = []
    transfer_out_2 = []
    speedup_2 = []
    speedup_2_noTrf = []

    kernel3 = []
    transfer_in_3 = []
    transfer_out_3 = []
    speedup_3 = []
    speedup_3_noTrf = []

    kernel4 = []
    transfer_in_4 = []
    transfer_out_4 = []
    speedup_4 = []
    speedup_4_noTrf = []

    kernel5 = []
    transfer_in_5 = []
    transfer_out_5 = []
    speedup_5 = []
    speedup_5_noTrf = []

    counter = 0
    for line in file:
        counter += 1
        if counter == 1:
            pass
        elif counter > 1 and (counter - 1) % 7 == 0:
            pass
        else:
            s = str.split(line)
            # cpu
            if (counter - 2) % 7 == 0:
                cpu.append(float(s[0]))

            # kernel 1
            if (counter - 3) % 7 == 0:
                kernel1.append(float(s[2]))
                transfer_in_1.append(float(s[3]))
                transfer_out_1.append(float(s[4]))
                speedup_1.append(float(s[5]))
                speedup_1_noTrf.append(float(s[6]))
            # kernel 2
            if (counter - 4) % 7 == 0:
                kernel2.append(float(s[2]))
                transfer_in_2.append(float(s[3]))
                transfer_out_2.append(float(s[4]))
                speedup_2.append(float(s[5]))
                speedup_2_noTrf.append(float(s[6]))
            # kernel 3
            if (counter - 5) % 7 == 0:
                kernel3.append(float(s[2]))
                transfer_in_3.append(float(s[3]))
                transfer_out_3.append(float(s[4]))
                speedup_3.append(float(s[5]))
                speedup_3_noTrf.append(float(s[6]))

            # kernel 4
            if (counter - 6) % 7 == 0:
                kernel4.append(float(s[2]))
                transfer_in_4.append(float(s[3]))
                transfer_out_4.append(float(s[4]))
                speedup_4.append(float(s[5]))
                speedup_4_noTrf.append(float(s[6]))
            # kernel 5
            if (counter - 7) % 7 == 0:
                kernel5.append(float(s[2]))
                transfer_in_5.append(float(s[3]))
                transfer_out_5.append(float(s[4]))
                speedup_5.append(float(s[5]))
                speedup_5_noTrf.append(float(s[6]))

        arr = []
        for i in range(len(kernel1)):
            arr.append(i)

    # Plot
    plt.yscale("log", basey=10, nonposy="clip")
    plt.plot(arr, kernel1, "r-")
    plt.plot(arr, kernel2, "b-")
    plt.plot(arr, kernel3, "g-")
    plt.plot(arr, kernel4, "m-")
    plt.plot(arr, kernel5, "k-")
    plt.plot(arr, cpu, "k--")
    plt.title("GPU time for Laplace-filtering ({0:s})".format(filename))
    plt.xlabel("Test number")
    plt.ylabel("log Milliseconds")
    plt.savefig("outputs/{0:s}_GPU_times.png".format(filename))
    plt.clf()

    plt.yscale("log", basey=10, nonposy="clip")
    plt.plot(arr, speedup_1, "r-")
    plt.plot(arr, speedup_2, "b-")
    plt.plot(arr, speedup_3, "g-")
    plt.plot(arr, speedup_4, "m-")
    plt.plot(arr, speedup_5, "k-")
    plt.title("GPU speedup by Laplace-filtering method ({0:s})".format(filename))
    plt.xlabel("Test number")
    plt.ylabel("log Speedup")
    plt.savefig("outputs/{0:s}_GPU_speedup.png".format(filename))
    plt.clf()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', help="Enter filename in the outputs folder")
args = parser.parse_args()
generate_results(args.filename)
