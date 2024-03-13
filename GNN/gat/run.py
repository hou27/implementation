import enum

# 3 different model training/eval phases used in train.py
class LoopPhase(enum.Enum):
    TRAIN = 0,
    VAL = 1,
    TEST = 2


# Global vars used for early stopping. After some number of epochs (as defined by the patience_period var) without any
# improvement on the validation dataset (measured via accuracy metric), we'll break out from the training loop.
BEST_VAL_ACC = 0
BEST_VAL_LOSS = 0

import time
import torch
import torch.nn as nn
from torch.optim import Adam

from gat import GAT

def train_gat(time_start, dataset, train_range, val_range, test_range, num_input_features, num_classes):
    global BEST_VAL_ACC, BEST_VAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: load the graph data
    node_features = dataset[0].x.to(device)
    node_labels = dataset[0].y.to(device)
    edge_index = dataset[0].edge_index.to(device)

    # Indices that help us extract nodes that belong to the train/val and test splits
    train_indices = torch.arange(train_range[0], train_range[1], dtype=torch.long, device=device)
    val_indices = torch.arange(val_range[0], val_range[1], dtype=torch.long, device=device)
    test_indices = torch.arange(test_range[0], test_range[1], dtype=torch.long, device=device)

    # Step 2: prepare the model
    gat = GAT(num_input_features, num_classes, add_skip_connection=True, bias=True, dropout=0.6).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=0.005, weight_decay=0.0005)

    node_dim = 0  # this will likely change as soon as I add an inductive example (Citeseer is transductive)
    # because in inductive case, the node features will already be shaped as (N, FIN) and not (E, FIN) as they are here
    # 즉, induction의 경우, node_features의 shape이 (N, FIN)이고, transduction의 경우, (E, FIN)이다.

    train_labels = node_labels.index_select(node_dim, train_indices)
    val_labels = node_labels.index_select(node_dim, val_indices)
    test_labels = node_labels.index_select(node_dim, test_indices)

    # node_features shape = (N, FIN), edge_index shape = (2, E)
    graph_data = (node_features, edge_index)  # I pack data into tuples because GAT uses nn.Sequential which requires it

    def main_loop(phase, epoch=0):
        global BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.
        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        # Depending on the current phase, we'll be working with different indices and labels
        if phase == LoopPhase.TRAIN:
            node_indices = train_indices
        elif phase == LoopPhase.VAL:
            node_indices = val_indices
        else:
            node_indices = test_indices

        if phase == LoopPhase.TRAIN:
            node_labels = train_labels
        elif phase == LoopPhase.VAL:
            node_labels = val_labels
        else:
            node_labels = test_labels

        # Do a forwards pass and extract only the relevant node scores (train/val or test ones)
        # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        # shape = (N, C) where N is the number of nodes in the split (train/val/test) and C is the number of classes
        nodes_unnormalized_scores = gat(graph_data)[0].index_select(node_dim, node_indices)

        # Example: let's take an output for a single node on Citeseer - it's a vector of size 7 and it contains unnormalized
        # scores like: V = [-1.393,  3.0765, -2.4445,  9.6219,  2.1658, -5.5243, -4.6247]
        # What PyTorch's cross entropy loss does is for every such vector it first applies a softmax, and so we'll
        # have the V transformed into: [1.6421e-05, 1.4338e-03, 5.7378e-06, 0.99797, 5.7673e-04, 2.6376e-07, 6.4848e-07]
        # secondly, whatever the correct class is (say it's 3), it will then take the element at position 3,
        # 0.99797 in this case, and the loss will be -log(0.99797). It does this for every node and applies a mean.
        # You can see that as the probability of the correct class for most nodes approaches 1 we get to 0 loss! <3
        loss = loss_fn(gat(graph_data)[0].index_select(node_dim, node_indices), node_labels)

        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights

        # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
        # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
        class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
        accuracy = torch.sum(torch.eq(class_predictions, node_labels).long()).item() / len(node_labels)

        if phase == LoopPhase.VAL:
            if epoch % 10 == 0:
                print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val acc={accuracy}')

            # The "patience" logic - should we break out from the training loop? If either validation acc keeps going up
            # or the val loss keeps going down we won't stop
            if accuracy > BEST_VAL_ACC or loss.item() < BEST_VAL_LOSS:
                BEST_VAL_ACC = max(accuracy, BEST_VAL_ACC)  # keep track of the best validation accuracy so far
                BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)
                PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
            else:
                PATIENCE_CNT += 1  # otherwise keep counting

            if PATIENCE_CNT >= 100:
                raise Exception('Stopping the training, the universe has no more patience for this training.')

        else:
            return accuracy  # in the case of test phase we just report back the test accuracy

    BEST_VAL_ACC, BEST_VAL_LOSS = [0, 0]  # reset vars used for early stopping

    # Step 4: Start the training procedure
    for epoch in range(100000):
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, epoch=epoch)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

    test_acc = main_loop(phase=LoopPhase.TEST)
    print(f'Test accuracy = {test_acc}')