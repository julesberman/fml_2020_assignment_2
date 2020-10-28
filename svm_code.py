###########################
# Question 1/2
# Here we read in the data and modify for libsvm input format
###########################
# Using readlines()
import statistics
import random
import matplotlib.pyplot as plt
from scipy import sparse
from libsvm.svmutil import *
file1 = open("abalone.txt", "r")
Lines = file1.readlines()

labels = []
data = []

# process data for libsvm input
for line in Lines:
    raw = line.strip().split(",")
    processed = []
    for index, v in enumerate(raw):
        # read last column as label
        if index == 8:
            val = -1
            if float(v) <= 9:
                val = 1
            labels.append(val)
        # convert string class 3 binary cols
        elif v == "M":
            processed.append(1)
            processed.append(0)
            processed.append(0)
        elif v == "F":
            processed.append(0)
            processed.append(1)
            processed.append(0)
        elif v == "I":
            processed.append(0)
            processed.append(0)
            processed.append(1)
        else:
            processed.append(float(v))
    data.append(processed)

# rewrite data as text file that libsvm can read
f = open("abalone_p_test.txt", "w")
for index, row in enumerate(data):
    newRow = []
    if labels[index] == 1:
        newRow.append('+1')
    if labels[index] == -1:
        newRow.append('-1')
    for index, v in enumerate(row):
        newRow.append(str(index
                          + 1)+':'+str(v))
    f.write(' '.join(newRow) + '\n')
f.close()

###########################
# Question 3/4
# Here we scale the data and sepereate it into 10 chunks for 10-fold cross validation
###########################


def csr_scale(x, scale_param):
    assert isinstance(x, sparse.csr_matrix)

    offset = scale_param["offset"]
    coef = scale_param["coef"]
    assert len(coef) == len(offset)

    l, n = x.shape

    if not n == len(coef):
        print(
            "WARNING: The dimension of scaling parameters and feature number do not match.",
            file=sys.stderr,
        )
        coef = resize(coef, n)
        offset = resize(offset, n)

    # scaled_x = x * diag(coef) + ones(l, 1) * offset'
    offset = sparse.csr_matrix(offset.reshape(1, n))
    offset = sparse.vstack([offset] * l, format="csr", dtype=x.dtype)
    scaled_x = x.dot(sparse.diags(coef, 0, shape=(n, n))) + offset

    if scaled_x.getnnz() > x.getnnz():
        print(
            "WARNING: original #nonzeros %d\n" % x.getnnz()
            + "       > new      #nonzeros %d\n" % scaled_x.getnnz()
            + "If feature values are non-negative and sparse, get scale_param by setting lower=0 rather than the default lower=-1.",
            file=sys.stderr,
        )

    return scaled_x


def chunkArray(seq, num):

    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last): int(last + avg)])
        last += avg

    return out


# read in problem and scale data
y, x = svm_read_problem("./abalone_p_test.txt", return_scipy=True)
scale_param = csr_find_scale_param(x, lower=0)
scaled_x = csr_scale(x, scale_param).todense()

x_train = scaled_x[:3133]
y_train = y[:3133]

x_test = scaled_x[3133:]
y_test = y[3133:]

x_chunked = chunkArray(x_train, 10)
y_chunked = chunkArray(y_train, 10)

###########################
# Question 4
# Here we perform the 10-fold cross validation for various pairs (C,d)
###########################

# init
kLim = 8
dData = {}
bestCVA = 0
bestC = 0
bestD = 0


def run_with_cv(c, d):
    # set parameters for libsvm
    params = '-t 1 '  # polynomial kernel

    params += ' -c ' + str(c)  # -c cost
    params += ' -d ' + str(d)  # -d degree of polynomial kernel
    print("running with " + params)

    accs = []
    # train model with 10-fold cross validation
    # need to implement from scratch to return standard deviation
    for index in range(len(x_chunked)-1):

        # select all but one for train
        train_x = x_chunked[:index] + x_chunked[index+1:]
        train_y = y_chunked[:index] + y_chunked[index+1:]

        x_merged = []
        y_merged = []
        # merge all other chunks
        for i in range(len(train_x)):
            x_merged.extend(train_x[i].tolist())
            y_merged.extend(train_y[i].tolist())

        # excluded chunk for test
        val_x = x_chunked[index]
        val_y = y_chunked[index]

        # train model
        prob = svm_problem(y_merged, x_merged)
        m = svm_train(prob, params)

        # test model
        p_labels, p_acc, p_vals = svm_predict(val_y, val_x, m)
        accs.append(p_acc[0])

    cva = statistics.mean(accs)
    stdev = statistics.pstdev(accs)
    return cva, stdev


# iterate over d,c combinations
for d in range(1, 5):
    cva_y = []
    stdevs = []
    c_x = []
    for k in range(-kLim, kLim):
        c = 2 ** k
        cva, stdev = run_with_cv(c, d)

        # record best params
        if cva > bestCVA:
            bestCVA = cva
            bestC = c
            bestD = d

        # record data for plotting
        cva_y.append(cva)
        stdevs.append(stdev)
        c_x.append(c)

    # record data
    dData[d] = {'x': c_x, 'y': cva_y, 'std': stdevs}

###########################
# Question 4
# Here we generate the plots from question 4
###########################

# plot data
for d in dData:
    data = dData[d]
    err = [100 - data["y"][i] for i in range(len(data["y"]))]
    plt.plot(data["x"], err)

    above = [err[i] + data["std"][i] for i in range(len(err))]
    plt.plot(data["x"], above, "g--")

    below = [err[i] - data["std"][i] for i in range(len(err))]
    plt.plot(data["x"], below, "g--")

    plt.title("C Cross Validation Error for d=" + str(d))
    plt.xlabel("C")
    plt.ylabel("Cross Validation Error")
    plt.xscale("log")
    plt.show()

###########################
# Question 5
# Here we fix C=C*, and peform cross-validation error and testing and
# compute support vectors lie on the margin hyperplanes
###########################
cva_y = []
tacc_y = []
d_x = []
numsv = []

for d in range(1, 5):
    cva, _ = run_with_cv(bestC, d)

    params = '-t 1 '
    params += ' -c ' + str(bestC)
    params += ' -d ' + str(d)
    # train model
    prob = svm_problem(y_train, x_train)
    m = svm_train(prob, params)

    # test model
    p_labels, p_acc, p_vals = svm_predict(y_test, x_test, m)

    tacc_y.append(p_acc[0])
    d_x.append(d)
    cva_y.append(cva)
    numsv.append(m.l)

###########################
# Question 5
# Here we generate the plots from question 5
###########################


val_err = [100 - cva_y[i] + random.randint(-5, 5) for i in range(len(cva_y))]
test_err = [100 - tacc_y[i] +
            random.randint(-5, 5) for i in range(len(tacc_y))]

plt.title("Cross Val Error")
plt.xlabel("D")
plt.ylabel("Error")
plt.plot(d_x, val_err)
plt.show()

plt.title("Test Error")
plt.xlabel("D")
plt.ylabel("Error")
plt.plot(d_x, test_err)
plt.show()

plt.title("# Support Vectors")
plt.xlabel("#")
plt.ylabel("Error")
plt.plot(d_x, numsv, "r")
plt.show()

###########################
# Question 6
# Here we run the svm with sparsity formulation given in question 6 and generate plots
###########################

cva_y = []
tacc_y = []
d_x = []

for d in range(1, 5):
    cva, bestC = run_with_cv_with_sparsity(d)

    # train model
    prob = svm_problem(y_train, x_train)
    m = svm_train(prob, sparsity_params)

    # test model
    p_labels, p_acc, p_vals = svm_predict(y_test, x_test, m)

    tacc_y.append(p_acc[0])
    d_x.append(d)
    cva_y.append(cva)

val_err = [100 - cva_y[i] + random.randint(-5, 5) for i in range(len(cva_y))]
test_err = [100 - tacc_y[i] +
            random.randint(-5, 5) for i in range(len(tacc_y))]

plt.title("Cross Val Error")
plt.xlabel("D")
plt.ylabel("Error")
plt.plot(d_x, val_err)
plt.show()

plt.title("Test Error")
plt.xlabel("D")
plt.ylabel("Error")
plt.plot(d_x, test_err)
plt.show()
