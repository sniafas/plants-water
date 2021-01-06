import tensorflow_addons as tfa
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, Precision, Recall, AUC
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


def losses_and_metrics(num_classes):
    """
    Define loss and metrics
    Loss: Categorical Crossentropy
    Metrics: (Train, Validation) Accuracy, Precision, Recall, F1

    Returns:
        loss, train loss, train acc, valid loss, valid acc, precision, recall, auc, f1
    """

    loss_fn = CategoricalCrossentropy(from_logits=True)
    train_loss = Mean(name='train_loss')
    train_accuracy = CategoricalAccuracy('train_accuracy')

    valid_loss = Mean(name='valid_loss')
    valid_accuracy = CategoricalAccuracy('valid_accuracy')

    precision = Precision(name='precision')
    recall = Recall(name='recall')
    auc = AUC(name='auc')
    f1_train = tfa.metrics.F1Score(num_classes=num_classes, average='macro')
    f1_loss = tfa.metrics.F1Score(num_classes=num_classes, average='macro')

    return loss_fn, train_loss, train_accuracy, valid_loss, valid_accuracy, precision, recall, auc, f1_train, f1_loss
