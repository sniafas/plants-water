from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD, Adadelta, Nadam

def get_optimizer(learning_rate, opt_name):
    '''Select optimizer method

    Arguments:
        learning_rate: float, learning value
        opt_name: str, optimizer name (adam, nadam, rms, adagrad, sgd, adadelta)

    Returns:
        optimizer object
    '''
    if opt_name == 'adam':
        optimizer = Adam(learning_rate)
    elif opt_name == 'nadam':
        optimizer = Nadam(learning_rate)
    elif opt_name == 'rms':
        optimizer = RMSprop(learning_rate)
    elif opt_name == 'adagrad':
        optimizer = Adagrad(learning_rate)
    elif opt_name == 'sgd':
        optimizer = SGD(0.01, nesterov=True)
    elif opt_name == 'adadelta':
        optimizer = Adadelta(learning_rate, rho=0.45)

    return optimizer