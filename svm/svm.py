import random
import csv


import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

def get_data(csv_path):
    input = []
    labels = []
    with open(csv_path, 'r') as csv_file:
        train_reader = csv.reader(csv_file)
        for sample in train_reader:
            example, label = sample[: -1], sample[-1]
            input.append([float(ex) for ex in example])
            labels.append(1 if int(label)==1 else -1)
        
    assert len(labels)==len(input)
    return input, labels

def get_lr_1(yt, t): # a = 2
    return yt/(1+ 0.5*yt*t)

def get_lr_2(yt, t):
    return yt/(1+ t)

def train(samples, labels, c, lr_func, epochs=100):

    num_samples = len(samples)
    # Initialize the SVM paramerters
    w = np.asarray([0 for _ in range(len(samples[0]))])
    b = 0
    yt = 1 #gamma_0

    sample_indx = list(range(num_samples))
    for t in range(1, epochs+1): 
        yt = lr_func(yt, t) # calculate new gamma value for this epoch
        random.shuffle(sample_indx) # shuffles the index 

        for i in sample_indx: # loop over samples and labels
            sample = samples[i]
            label = labels[i]

            y_pred = predict(w, b, sample)

            if label!=y_pred:
                w_update = w * (-1*yt) + (sample * (c*num_samples*yt*label))
                w = w + w_update
            else:
                w = w * (1-yt)
                b += yt*label 
    return w, b  

def predict(w, b, sample)-> int: # predict classification for svm
    return np.sign(np.dot(w.T, sample) - b)


def error(w, b, test_samples, test_labels):
    acc=0
    for sample, y in zip(test_samples, test_labels):
        pred = predict(w, b, sample)
        if pred!=y:
            acc += 1
    return acc/len(test_samples)

def q_2_a(train_samples, train_labels, test_samples, test_labels):
    print(f"For Question 2 (a)")
    epochs = 100
    C_values = [100/873, 500/873, 700/873] 
    lr_func= get_lr_1 # just change the lr_1 to lr_2 for q2
    for c in C_values:
        w, b = train(train_samples, train_labels, c, lr_func, epochs)
        train_error = error(w, b, train_samples, train_labels)
        print(f"Train error for {c} is {train_error}")
        test_error = error(w, b, test_samples, test_labels)
        print(f"Test error for {c} is {test_error}\n")

def q_2_b(train_samples, train_labels, test_samples, test_labels):
    print(f"For Question 2 (b)")
    epochs = 100
    C_values = [100/873, 500/873, 700/873] 
    lr_func= get_lr_2 # just change the lr_1 to lr_2 for q2
    for c in C_values:
        w, b = train(train_samples, train_labels, c, lr_func, epochs)
        train_error = error(w, b, train_samples, train_labels)
        print(f"Train error for {c} is {train_error}")
        test_error = error(w, b, test_samples, test_labels)
        print(f"Test error for {c} is {test_error}\n")

def q_2_c(train_samples, train_labels, test_samples, test_labels):
    print(f"For Question 2 (a)")
    epochs = 100
    C_values = [100/873, 500/873, 700/873]
    for c in C_values:
        w, b = train(train_samples, train_labels, c, get_lr_1, epochs)
        train_error_1 = error(w, b, train_samples, train_labels)
        test_error_1 = error(w, b, test_samples, test_labels)

        w, b = train(train_samples, train_labels, c, get_lr_2, epochs)
        train_error_2 = error(w, b, train_samples, train_labels)
        test_error_2 = error(w, b, test_samples, test_labels)
        print(f"Train error for {c} is for (a) {train_error_1} (b) {train_error_2} and difference is {train_error_1-train_error_2}")
        print(f"Test error for {c} is for (a) {test_error_1} (b) {test_error_2} and difference is {test_error_1-test_error_2}")



def q_3_a(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    def objective(alpha, X, y):
        # SVM dual objective function
        return 0.5 * np.sum(alpha**2) - np.sum(alpha * y * (X @ X.T @ y))

    def constraint(alpha):
        # Equality constraint: sum(alpha * y) = 0
        return np.sum(alpha * y_train)
    
    def predict(X, alpha, support_vectors, support_vector_labels):
        # SVM decision function
        return np.sign(np.sum(alpha * support_vector_labels * (X @ support_vectors.T), axis=1))
    
    # Initial guess for alpha
    alpha_init = np.zeros(len(x_train))

    # Equality constraint dictionary
    eq_cons = {'type': 'eq', 'fun': constraint}

    C_values = [100/873, 500/873, 700/873]
    for c in C_values:
        # Bounds for alpha (0 <= alpha <= C, where C is a positive constant)
        bounds = [(0, c) for _ in range(len(x_train))]

        # Solve the SVM dual problem using SLSQP
        result = minimize(objective, alpha_init, args=(x_train, y_train), 
                        method='SLSQP', bounds=bounds, constraints=eq_cons)

        # Get alphas that statisfy the constraint
        alpha_optimal = result.x

        # Support vectors and labels
        support_vectors = x_train
        support_vector_labels = y_train

        # Make predictions on the test set
        y_pred = predict(x_test, alpha_optimal, support_vectors, support_vector_labels)

        # Evaluate the model
        err = np.mean(y_pred != y_test)
        print(f"For Hyperparameter C {c} Error: {err}")


def q_3_b(x_train, y_train, x_test, y_test):

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    def gaussian_kernel(xi, xj, gamma):
        return np.exp(-gamma * np.linalg.norm(xi - xj)**2)

    def kernel_matrix(X, X2, gamma):
        n_samples = X.shape[0]
        n2_samples = X2.shape[0]
        K = np.zeros((n_samples, n2_samples))
        for i in range(n_samples):
            for j in range(n2_samples):
                K[i, j] = gaussian_kernel(X[i], X2[j], gamma)
        return K

    def objective(alpha, y, K):
        # SVM dual objective function with Gaussian kernel
        return 0.5 * np.sum(np.dot(alpha, np.dot(K, alpha))) - np.sum(alpha * y)

    def constraint(alpha):
        # Equality constraint: sum(alpha * y) = 0
        return np.sum(alpha * y_train)
    
    def predict( alpha, support_vector_labels, K):
        # SVM decision function
        return np.sign(np.sum(alpha* support_vector_labels * K, axis=1))

    # Equality constraint dictionary
    eq_cons = {'type': 'eq', 'fun': constraint}

    gammas = [0.1, 0.5, 1, 5, 100]
    C_values = [100/873, 500/873, 700/873]
    for gamma in gammas:
        for c in C_values:
            # Initial guess for alpha
            alpha_init = np.zeros(len(x_train))
            
            # Bounds for alpha (0 <= alpha <= C, where C is a positive constant)
            bounds = [(0, c) for _ in range(len(x_train))]

            K_train = kernel_matrix(x_train,x_train, gamma)

            # Solve the SVM dual problem using SLSQP
            result = minimize(objective, alpha_init, args=(K_train, y_train), 
                            method='SLSQP', bounds=bounds, constraints=eq_cons)

            alpha_optimal = result.x

            # Support vectors have non-zero alpha values
            support_vectors = x_train
            support_vector_labels = y_train

            # Make predictions on the test set
            K = kernel_matrix(x_train,support_vectors, gamma)
            y_pred = predict( alpha_optimal, support_vector_labels, K)

            # Evaluate the model
            err = np.mean(y_pred != y_train)
            print(f"For C {c} and Gamma {gamma} Training Error: {err}")

            # Make predictions on the test set
            K = kernel_matrix(x_test,support_vectors, gamma)
            y_pred = predict( alpha_optimal, support_vector_labels, K)

            # Evaluate the model
            err = np.mean(y_pred != y_test)
            print(f"For C {c} and Gamma {gamma} Test Error: {err}")


#question q3c
def q_3_c(x_train, y_train, x_test, y_test):

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    def gaussian_kernel(xi, xj, gamma):
        return np.exp(-gamma * np.linalg.norm(xi - xj)**2)

    def kernel_matrix(X, X2, gamma):
        n_samples = X.shape[0]
        n2_samples = X2.shape[0]
        K = np.zeros((n_samples, n2_samples))
        for i in range(n_samples):
            for j in range(n2_samples):
                K[i, j] = gaussian_kernel(X[i], X2[j], gamma)
        return K

    def objective(alpha, y, K):
        # SVM dual objective function with Gaussian kernel
        return 0.5 * np.sum(np.dot(alpha, np.dot(K, alpha))) - np.sum(alpha * y)

    def constraint(alpha):
        # Equality constraint: sum(alpha * y) = 0
        return np.sum(alpha * y_train)
    
    def predict( alpha, support_vector_labels, K):
        # SVM decision function
        return np.sign(np.sum(alpha* support_vector_labels * K, axis=1))

    # Equality constraint dictionary
    eq_cons = {'type': 'eq', 'fun': constraint}

    gammas = [0.1, 0.5, 1, 5, 100]
    c = 500/873
    old_support_vec = None
    for gamma in gammas:
        # Initial guess for alpha
        alpha_init = np.zeros(len(x_train))
        
        # Bounds for alpha (0 <= alpha <= C, where C is a positive constant)
        bounds = [(0, c) for _ in range(len(x_train))]

        K_train = kernel_matrix(x_train,x_train, gamma)

        # Solve the SVM dual problem using SLSQP
        result = minimize(objective, alpha_init, args=(K_train, y_train), 
                        method='SLSQP', bounds=bounds, constraints=eq_cons)

        alpha_optimal = result.x

        # Support vectors have non-zero alpha values
        support_vectors = x_train
        support_vector_labels = y_train

        if old_support_vec is not None:
            diff = np.zeros_like(support_vectors)
            diff = diff[old_support_vec==support_vectors]
            print(f"The common elements with previous support vectors is {len(diff.nonzero())}")
        old_support_vec = support_vectors

if __name__=="__main__":
    train_samples, train_labels = get_data("bank-note/train.csv") 
    test_samples, test_labels = get_data("bank-note/test.csv")

    x_train = np.asarray(train_samples)
    y_train = np.asarray(train_labels)
    x_test = np.asarray(test_samples)
    y_test = np.asarray(test_labels)      

    q_2_a(x_train, y_train, x_test, y_test)
    
    print("Question 2(b)")
    q_2_b(x_train, y_train, x_test, y_test)
    q_2_c(x_train, y_train, x_test, y_test)

    print(f"\nFor Question 3\n")
    print("Question 3 (a)")
    q_3_a(x_train, y_train, x_test, y_test)
    print("\nQuestion 3 (b)")
    q_3_b(x_train, y_train, x_test, y_test)
    print("For Question 3c:")
    q_3_c(x_train, y_train, x_test, y_test)
