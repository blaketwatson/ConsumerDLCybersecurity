import torch
import os
import sys
import pandas
import csv
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from random import randint
from random import seed
import time

Dict1 = {
    "tcp": 1,
    "udp": 2,
    "icmp": 3
}

Dict2 = {
    "private": 1,
    "domain_u": 2,
    "http": 3,
    "smtp": 4,
    "ftp_data": 5,
    "ftp": 6,
    "eco_i": 7,
    "other": 8,
    "auth": 9,
    "ecr_i": 10,
    "IRC": 11,
    "X11": 12,
    "finger": 13,
    "time": 14,
    "domain": 15,
    "telnet": 16,
    "pop_3": 17,
    "ldap": 18,
    "login": 19,
    "name": 20,
    "ntp_u": 21,
    "http_443": 22,
    "sunrpc": 23,
    "printer": 24,
    "systat": 25,
    "tim_i": 26,
    "netstat": 27,
    "remote_job": 28,
    "link": 29,
    "urp_i": 30,
    "sql_net": 31,
    "bgp": 32,
    "pop_2": 33,
    "tftp_u": 34,
    "uucp": 35,
    "imap4": 36,
    "pm_dump": 37,
    "nnsp": 38,
    "courier": 39,
    "daytime": 40,
    "iso_tsap": 41,
    "echo": 42,
    "discard": 43,
    "ssh": 44,
    "whois": 45,
    "mtp": 46,
    "gopher": 47,
    "rje": 48,
    "ctf": 49,
    "supdup": 50,
    "csnet_ns": 51,
    "uucp_path": 52,
    "nntp": 53,
    "netbois_ns": 54,
    "netbois_dgm": 55,
    "netbois_ssn": 56,
    "vmnet": 57,
    "Z39_50": 58,
    "exec": 59,
    "shell": 60,
    "efs": 61,
    "klogin": 62,
    "kshell": 63,
    "icmp": 64
}

Dict3 = {
    "SF": 1,
    "RSTR": 2,
    "S1": 3,
    "REJ": 4,
    "S3": 5,
    "RSTO": 6,
    "S0": 7,
    "S2": 8,
    "RSTOS0": 9,
    "SH": 10,
    "OTH": 11
}

DictLabels = {
    "normal": 1,
    "snmpgetattack": 2,
    "named": 3,
    "xlock": 4,
    "smurf": 5,
    "ipsweep": 6,
    "multihop": 7,
    "xsnoop": 8,
    "sendmail": 9,
    "guess_passwd": 10,
    "saint": 11,
    "buffer_overflow": 12,
    "portsweep": 13,
    "pod": 14,
    "apache2": 15,
    "phf": 16,
    "udpstorm": 17,
    "warezmaster": 18,
    "perl": 19,
    "satan": 20,
    "xterm": 21,
    "mscan": 22,
    "processtable": 23,
    "ps": 24,
    "nmap": 25,
    "rootkit": 26,
    "neptune": 27,
    "loadmodule": 28,
    "imap": 29,
    "back": 30,
    "httptunnel": 31,
    "worm": 32,
    "mailbomb": 33,
    "ftp_write": 34,
    "teardrop": 35,
    "land": 36,
    "sqlattack": 37,
    "snmpguess": 38
}

DEBUG = False
# Class Definition
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_predicted = self.linear2(h_relu)
        return y_predicted

# Class definition for basic convolutional net
# Note: this was abandoned in the final research paper and is not functional
class ConvolutionalNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(ConvolutionalNet, self).__init__()
        # D_in input channel (dataset), split out to 6x channels, 5 by 5 convolutional square
        self.conv1 = torch.nn.Conv2d(D_in, D_in*6, 10)
        self.conv2 = torch.nn.Conv2d(D_in*6, D_in*16, 10)
        # Take D_in * 16 * 5 * 5 and throw it into a linear connection
        self.linear1 = torch.nn.Linear(D_in*16*10*10, D_in*16)
        self.linear2 = torch.nn.Linear(D_in*16,D_in)
        self.linear3 = torch.nn.Linear(D_in,int(D_in/2))
        self.linear4 = torch.nn.Linear(int(D_in/2),int(D_in/4))
        self.linear5 = torch.nn.Linear(int(D_in/4),D_out)
        
    def foward(self, x):
        # Convolutional Pass
        x = F.max_pool2d(F.relu(self.conv1(x)), 5)
        x = F.max_pool2d(F.relu(self.conv2(x)), 5)
        # Linear Pass
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x

N = 4898431 # Actual Batch Size (Uncomment for real deal)
#N = 5000 # Test Batch Size (Comment for real deal)
D_in = 41 # Input Dimension
D_out = 1 # Output Dimension
H = 30 # Hidden Dimension 1
#H2 = 15 # Hidden Dimension 2

# Create tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Import data into x (input data)
print("Copying Standard Data...")
lineIterator = 0
collumnIterator = 0
with open("~/ResearchData/Labels.csv","r") as normalData:
    reader = csv.reader(normalData)
    with tqdm(total=N) as pbar:
        for line in reader:
            copy = line.copy()
            for item in copy:
                x[lineIterator][collumnIterator] = int(item)
                pbar.update(1)
                collumnIterator += 1
            collumnIterator = 0
            lineIterator += 1
if DEBUG:
    test = input("Press Enter to Continue...")

# Import data into y (output data)
print("Copying Labbelled Data...")
fileIterator = 0
with open("~/ResearchData/Labels.csv","r") as LabelledData:
    reader = csv.reader(LabelledData)
    with tqdm(total=N) as pbar:
        for line in reader:
            y[fileIterator][0] = int(line[0])
            fileIterator += 1
            pbar.update(1)
            
if DEBUG:
    test = input("Press Enter to Continue...")
    
# Optional Label Verification (pretty simplistic but it gets the job done)
if DEBUG:
    print("Verifying Labels...")
    zeroCounter = 0
    oneCounter = 0
    with tqdm(total=4898431) as pbar:
        for i in range(4898431):
            if y[i][0] == 1:
                oneCounter += 1
            elif y[i][0] == 0:
                zeroCounter += 1
            pbar.update(1)
    print("Zeroes:")
    print(zeroCounter)
    print("Ones:")
    print(oneCounter)
    verifiedZeroes = 972781
    verifiedOnes = 3925650
    if zeroCounter != verifiedZeroes or oneCounter != verifiedOnes:
        print("ERROR: Labels were not transferred properly.")
    test = input("Press enter to continue... ")

# Optional: Scramble data
# TODO: Test this. It currently has about a 2 hour runtime, and while it should work based on some smaller scale tests, it needs to be further tested to verify usability. It should be noted that this is a highly inefficient algorithm and is certaintly not the most efficient way to randomize. This was thrown together in about 60 seconds. It was not utilized in the research paper and is entirely optional.
if DEBUG:
    test = input("Randomize Tensors? [y/n]: ")
else:
    test = "n"
    
if test == "y":
    print("Randomizing Tensors...")
    usedNums = []
    newx = torch.randn(N, D_in)
    newy = torch.randn(N, D_out)
    newIterator = 0
    with tqdm(total=N) as pbar:
        while True:
            value = randint(0,N)
            if value not in usedNums:
                # Give new values to new tensors
                newx[newIterator] = x[value]
                newy[newIterator] = y[value]
                usedNums.append(value)
                pbar.update(1)
            if len(usedNums) >= N:
                break

    failedNums = []
    print("Verifying Randomization...")
    with tqdm(total=N) as pbar:
        for i in range(N):
            if i not in usedNums:
                failedNums.append(i)
            pbar.update(1)
    if len(failedNums) != 0:
        print("Some values were not copied...")
        print(failedNums)
    
    print("Verifying Labels...")
    zeroCounter = 0
    oneCounter = 0
    with tqdm(total=4898431) as pbar:
        for i in range(4898431):
            if y[i][0] == 1:
                oneCounter += 1
            elif y[i][0] == 0:
                zeroCounter += 1
            pbar.update(1)
    if zeroCounter != verifiedZeroes or oneCounter != verifiedOnes:
        print("ERROR: Labels were not transferred properly.")

if DEBUG:
    test = input("Press Enter to Continue...")

print("Splitting Data Set...")
# Dataset splitting
x1 = torch.randn(int(N/10) + 1, D_in)
x2 = torch.randn(int(N/10) + 1, D_in)
x3 = torch.randn(int(N/10) + 1, D_in)
x4 = torch.randn(int(N/10) + 1, D_in)
x5 = torch.randn(int(N/10) + 1, D_in)
x6 = torch.randn(int(N/10) + 1, D_in)
x7 = torch.randn(int(N/10) + 1, D_in)
x8 = torch.randn(int(N/10) + 1, D_in)
x9 = torch.randn(int(N/10) + 1, D_in)
x10 = torch.randn(int(N/10) + 1, D_in)
y1 = torch.randn(int(N/10) + 1, D_out)
y2 = torch.randn(int(N/10) + 1, D_out)
y3 = torch.randn(int(N/10) + 1, D_out)
y4 = torch.randn(int(N/10) + 1, D_out)
y5 = torch.randn(int(N/10) + 1, D_out)
y6 = torch.randn(int(N/10) + 1, D_out)
y7 = torch.randn(int(N/10) + 1, D_out)
y8 = torch.randn(int(N/10) + 1, D_out)
y9 = torch.randn(int(N/10) + 1, D_out)
y10 = torch.randn(int(N/10) + 1, D_out)

# I know this is extremely innefficient and can probaby be done better, but this should work for now.
iterator1 = 0
iterator2 = 0
iterator3 = 0
iterator4 = 0
iterator5 = 0
iterator6 = 0
iterator7 = 0
iterator8 = 0
iterator9 = 0
iterator10 = 0

with tqdm(total=N) as pbar:
    for i in range(N):
        if i % 10 == 0:
            x1[iterator1] = x[i]
            y1[iterator1] = y[i]
            iterator1 += 1
            pbar.update(1)
        if i % 10 == 1:
            x2[iterator2] = x[i]
            y2[iterator2] = y[i]
            iterator2 += 1
            pbar.update(1)
        if i % 10 == 2:
            x3[iterator3] = x[i]
            y3[iterator3] = y[i]
            iterator3 += 1
            pbar.update(1)
        if i % 10 == 3:
            x4[iterator4] = x[i]
            y4[iterator4] = y[i]
            iterator4 += 1
            pbar.update(1)
        if i % 10 == 4:
            x5[iterator5] = x[i]
            y5[iterator5] = y[i]
            iterator5 += 1
            pbar.update(1)
        if i % 10 == 5:
            x6[iterator6] = x[i]
            y6[iterator6] = y[i]
            iterator6 += 1
            pbar.update(1)
        if i % 10 == 6:
            x7[iterator7] = x[i]
            y7[iterator7] = y[i]
            iterator7 += 1
            pbar.update(1)
        if i % 10 == 7:
            x8[iterator8] = x[i]
            y8[iterator8] = y[i]
            iterator8 += 1
            pbar.update(1)
        if i % 10 == 8:
            x9[iterator9] = x[i]
            y9[iterator9] = y[i]
            iterator9 += 1
            pbar.update(1)
        if i % 10 == 9:
            x10[iterator10] = x[i]
            y10[iterator10] = y[i]
            iterator10 += 1
            pbar.update(1)

# Save all tensors into a list for easy access
xtensorList = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
ytensorList = [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10]

if DEBUG:
    test = input("Press Enter to Continue...")

print("Initializing Model...")
testModel = TwoLayerNet(D_in, H, D_out)
testConvModel = ConvolutionalNet(D_in, D_out)

# Create the loss function
print("Initializing Loss Function...")
loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn_printer = torch.nn.L1Loss(reduction='mean')

# Create the optimizer
# Notes:
# Should probably switch this over to ADAM as initial tests showed better results
print("Initializing Optimizer...")
optimizer = torch.optim.SGD(testModel.parameters(), lr=1e-6)

print("Entering Optimized Training Cycle...")
os.system("rm ~/ResearchData/lossOutput.txt")
os.system("touch ~/ResearchData/lossOutput.txt")

if DEBUG:
    runThroughs = input("Enter % of data to train: [10,20,30,40,50,60,70,80]: ")
    runThroughs = int((int(runThroughs) / 10))
else:
    runThroughs = 8
rangeRun = int(N/10) + 1
predictedLength = 2000
print("Higher Level Loops: " + str(runThroughs))
if DEBUG:
    minIncrease = input("Enter minimum required error decrease [0.1-0.5 recommended]: ")
else:
    minIncrease = 0.01
print("Minimum Required Error Decrease: " + str(minIncrease))

# NOTE: The 10,000 flat epoch trial code is not included here, as it is pretty simple given the following code.
        
with open("~/ResearchData/lossOutput.txt","w") as output:
    for i in range(runThroughs): # Run 1-8 times
        print("Running Runthrough " + str(i + 1) + "...")
        currError = 1
        prevError = 1
        # Do an initial two runs just to set up the initial errors
        y_predicted = testModel(xtensorList[i])
        loss = loss_fn(y_predicted, ytensorList[i])
        prevError = loss.item()
        output.write(str(0) + " " + str(loss.item()) + "\n")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_predicted = testModel(xtensorList[i])
        loss = loss_fn(y_predicted, ytensorList[i])
        currError = loss.item()
        output.write(str(1) + " " + str(loss.item()) + "\n")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iterator = 0
        pbar = tqdm(total=predictedLength)
        for k in range(predictedLength):
            # Ensure that at least 50 epochs are performed
            if float(prevError - currError) <= float(minIncrease) and iterator >= 50:
                predictedLength = 100
                pbar.close()
                break
                
            # Forward Pass
            y_predicted = testModel(xtensorList[i])

            # Loss calculation and output
            loss = loss_fn(y_predicted, ytensorList[i])
#            loss_printer = loss_fn_printer(y_predicted, ytensorList[i])
            loss_item = loss.item()
            output.write(str(k + 2) + " " + str(loss_item) + "\n")
            prevError = currError
            currError = loss_item

            optimizer.zero_grad() # Zero gradients
            loss.backward() # Run backwards pass
            optimizer.step() # Update weights

            pbar.update(1)
            iterator += 1
        pbar.close()

test = input("Press Enter to Continue...")


# Pretty basic evaluation cycle
flipFlop = 8
for t in range(500):

    time.sleep(1)
    y_predicted = testModel(xtensorList[flipFlop]) # Forward Pass
    loss = loss_fn_printer(y_predicted, ytensorList[flipFlop]) # Get Loss
    optimizer.zero_grad()
    print(str(t) + ": " + str(loss.item()))
    
    # Change which one we are using
    if flipFlop == 8:
        flipFlop = 9
    else:
        flipFlop = 8
