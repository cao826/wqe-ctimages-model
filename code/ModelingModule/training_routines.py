"""
Module Level Docstring
"""
import numpy as np
from torch import nn
import torch

def train_on_batch(batch, model, loss_fn, optimizer):
    """Trains the model on a batch of the data"""
    model.train()
    inputs, clinical_info_batch, labels = batch
    inputs = inputs.cuda()
    clinical_info_batch = clinical_info_batch.cuda()
    labels = labels.cuda()

    optimizer.zero_grad()
    outputs = model(inputs, clinical_info_batch)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss.item()

def print_training_loss(loss_list, epoch_number=None):
    """Print the mean batch loss on the training set"""
    mean_loss = sum(loss_list) / len(loss_list)
    if epoch_number:
        print(f"Training loss on epoch {epoch_number}: {mean_loss}")
    else:
        print(f"Training loss: {mean_loss}")

def train_one_epoch(model, training_loader,
                    loss_fn, optimizer, epoch_number=None):
    """Trains the model for one epoch"""
    losses = []
    for batch in training_loader:
        losses.append( train_on_batch(batch = batch,
                                      model = model,
                                      loss_fn = loss_fn,
                                      optimizer = optimizer))
    print_training_loss(losses, epoch_number)

def eval_one_batch(batch, model, loss_fn):
    """Evaluates the model on one batch of the validation dataloader"""
    model.eval()
    inputs, clinical_info, labels = batch
    inputs = inputs.cuda()
    clinical_info = clinical_info.cuda()
    labels = labels.cuda()
    outputs = model(inputs, clinical_info)
    loss = loss_fn(outputs, labels)
    return loss.item()

def eval_on_dataloader(model, val_loader, loss_fn, epoch_number=None):
    """Evaluates the model on the validation data"""
    losses = []
    for batch in val_loader:
        losses.append(
            eval_one_batch(
                batch = batch,
                model = model,
                loss_fn = loss_fn
            )
        )
    print_average_validation_loss(val_loss_lst=losses,
                                  epoch_number=epoch_number)

def print_average_validation_loss(val_loss_lst, epoch_number=None):
    """Prints the average batch loss on val set"""
    mean_validation_loss = np.array(val_loss_lst).mean()
    if epoch_number:
        print(f"Mean validation loss on epoch {epoch_number}: {mean_validation_loss}")
    else:
        print(f"Mean validation loss: {mean_validation_loss}")

def logits_to_preds_cross(logits_tensor):
    """Converts the class scores output by model to class probabilities"""
    probs = nn.Softmax(dim=1)(
        logits_tensor.detach()
        )
    preds = torch.argmax(
        probs,
        dim=1
    )
    return preds
