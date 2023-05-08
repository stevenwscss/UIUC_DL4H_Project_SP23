import os
import time
import psutil
import torch
import torch.nn as nn
import numpy as np
import DataProcessing as dp
from Hyperpara_config import Config
from PatientDataLoader import Data_Loader
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score


def run(data_set, FLAGS, isCONTENT=True, isRNN=False, isLSTM=False):
    def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
        assert inputs is not None
        if shuffle:
            indices = np.arange(len(inputs[0]))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs[0]) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield [torch.tensor(input[excerpt]) for input in inputs]


    class ThetaLayer(nn.Module):
        def __init__(self):
            super(ThetaLayer, self).__init__()

            self.l_1 = nn.Linear(N_VOCAB, FLAGS.n_hidden)
            self.l_2 = nn.Linear(FLAGS.n_hidden, FLAGS.n_hidden)
            self.mu = nn.Linear(MAX_LENGTH * FLAGS.n_hidden, FLAGS.n_topics)
            self.log_sigma = nn.Linear(MAX_LENGTH * FLAGS.n_hidden, FLAGS.n_topics)
            self.relu = nn.ReLU()

        def forward(self, x):
            l_1 = self.relu(self.l_1(x))
            l_2 = self.relu(self.l_2(l_1))
            mu = self.mu(l_2.view(FLAGS.batch_size, -1))
            log_sigma = self.log_sigma(l_2.view(FLAGS.batch_size, -1))
            sigma = torch.exp(log_sigma)
            klterm = torch.sum(sigma) - torch.sum(log_sigma) + torch.trace(torch.matmul(mu, torch.transpose(mu, 0, 1)))

            eps = torch.randn_like(mu)
            theta = torch.nn.functional.softplus(mu + eps * sigma)

            thetatile = theta.unsqueeze(1).repeat(1, MAX_LENGTH, 1)

            return thetatile, klterm

    class ElemwiseMergeLayer(nn.Module):
        def __init__(self, merge_function, **kwargs):
            super().__init__()
            self.merge_function = merge_function
            self.kwargs = kwargs
            
        def forward(self, x, y):
            return self.merge_function(x, y, **self.kwargs)

    class ContentModel(nn.Module):
        def __init__(self):
            super().__init__()

            # Embed layer
            self.l_embed= nn.Linear(N_VOCAB, FLAGS.embed_dim)

            # Forward GRU layer
            self.l_forward0 = nn.GRU(FLAGS.embed_dim, FLAGS.n_hidden, batch_first=True)

            # Theta layer
            self.l_theta = ThetaLayer()

            # Context layer
            self.l_B = nn.Linear(N_VOCAB, FLAGS.n_topics, bias=False)
            self.l_context = ElemwiseMergeLayer(torch.mul)

            # Output layer
            self.l_dense0 = nn.Linear(FLAGS.n_hidden, 1)
            self.l_dense1 = nn.Flatten(start_dim=1)
            self.l_dense = ElemwiseMergeLayer(torch.add)
            self.l_out0 = nn.Sigmoid()

        def forward(self, x, mask):
            # Input layer
            g = self.l_embed(x)

            # Mask layer
            l_mask = mask.unsqueeze(2)
            g = g * l_mask

            # Forward GRU layer
            g, _ = self.l_forward0(g)

            # Masked GRU layer
            g = g * l_mask

            # Theta layer
            t, kl_term = self.l_theta(x)
            self.kl_term = kl_term

            # Context layer
            c = self.l_B(x)
            c = self.l_context(c, t)
            c = torch.mean(c, dim=2)

            # Output layer
            o = self.l_dense0(g)
            o = self.l_dense1(o)
            o = self.l_dense(o, c)
            o = self.l_out0(o)
            o = o * mask + 0.000001
            return o


    def train(model, use_validation = False):
        min_valid_loss = np.inf

        print("Training...")

        for epoch in range(FLAGS.total_epoch):
            start_time = time.time()
            train_loss = 0.0
            train_batches = 0
            model.train()
            for batch in iterate_minibatches_listinputs([trainingAdmiSeqs, trainingLabels, trainingMask], FLAGS.batch_size,
                                                                shuffle=True):
                # Compute prediction and loss
                pred = model(batch[0], batch[2])
                loss = loss_fn(pred * batch[2], batch[1] * batch[2]) + model.kl_term

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / train_batches}')
            
            
            if use_validation == True:
                valid_loss = 0.0
                valid_batches = 0
                model.eval()
                for batch in iterate_minibatches_listinputs([validAdmiSeqs, validLabels, validMask, validLengths], 1, shuffle=False):
                    # Compute prediction and loss
                    pred = model(batch[0], batch[2])
                    loss = loss_fn(pred * batch[2], batch[1] * batch[2]) + model.kl_term
                    valid_loss += loss.item()
                    valid_batches += 1
                print(f'Epoch {epoch+1} \t\t Validation Loss: {valid_loss / valid_batches}')
                if min_valid_loss > valid_loss:
                    print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                    min_valid_loss = valid_loss
                        
                    # Saving State Dict
                    torch.save(model.state_dict(), 'saved_model.pth')
            print("Epoch {} of {} took {:.3f}s".format(epoch+1, Config.total_epoch, time.time() - start_time))
        if use_validation == True:
            model.load_state_dict(torch.load('saved_model.pth'))

    def test(model):
        print("Testing...")
        start_time=time.time()

        test_pred = torch.LongTensor()
        test_score = torch.Tensor()
        test_true = torch.LongTensor()
        model.eval()
        for batch in iterate_minibatches_listinputs([validAdmiSeqs, validLabels, validMask, validLengths], 1, shuffle=False):
            leng = batch[3][0]
            pred = model(batch[0], batch[2])
            pred = torch.flatten(pred)[:leng]
            true = torch.flatten(batch[1])[:leng]
            test_score = torch.cat((test_score, pred.detach().to('cpu')), dim=0)
            pred = (pred > FLAGS.threshold).int()
            test_pred = torch.cat((test_pred, pred.detach().to('cpu')), dim=0)
            test_true = torch.cat((test_true, true.detach().to('cpu')), dim=0)

        pr_auc = average_precision_score(test_true, test_score)
        roc_auc = roc_auc_score(test_true, test_score)
        acc = accuracy_score(test_true, test_pred)

        print("Test roc_auc:\t\t{:.6f}".format(roc_auc))
        print("Test pr_auc:\t\t{:.6f}".format(pr_auc))
        print("Test acc:\t\t{:.6f}".format(acc))
        print(f"Total time to test: {time.time()-start_time}")


    class RNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.kl_term = 0

            # Embed layer
            self.l_embed= nn.Linear(N_VOCAB, FLAGS.embed_dim)

            # Forward GRU layer
            self.l_forward0 = nn.GRU(FLAGS.embed_dim, FLAGS.n_hidden, batch_first=True)
            self.l_dense0 = nn.Linear(FLAGS.n_hidden, 1)
            self.l_dense1 = nn.Flatten(start_dim=1)
            self.l_out0 = nn.Sigmoid()

        def forward(self, x, mask):
            # Input layer
            g = self.l_embed(x)
            # Mask layer
            l_mask = mask.unsqueeze(2)
            g = g * l_mask
            # Forward GRU layer
            g, _ = self.l_forward0(g)
            g = g * l_mask
            o = self.l_dense0(g)
            o = self.l_dense1(o)
            o = self.l_out0(o)
            o = o * mask + 0.000001
            return o

    class ContentModel_LSTM(nn.Module):
        def __init__(self):
            super().__init__()

            # Embed layer
            self.l_embed= nn.Linear(N_VOCAB, FLAGS.embed_dim)

            # Forward LSTM layer
            self.l_forward0 = nn.LSTM(FLAGS.embed_dim, FLAGS.n_hidden, batch_first=True)

            # Theta layer
            self.l_theta = ThetaLayer()

            # Context layer
            self.l_B = nn.Linear(N_VOCAB, FLAGS.n_topics, bias=False)
            self.l_context = ElemwiseMergeLayer(torch.mul)

            # Output layer
            self.l_dense0 = nn.Linear(FLAGS.n_hidden, 1)
            self.l_dense1 = nn.Flatten(start_dim=1)
            self.l_dense = ElemwiseMergeLayer(torch.add)
            self.l_out0 = nn.Sigmoid()

        def forward(self, x, mask):
            # Input layer
            l = self.l_embed(x)

            # Mask layer
            l_mask = mask.unsqueeze(2)
            l = l * l_mask

            # Forward LSTM layer
            l, _ = self.l_forward0(l)

            # Masked LSTM layer
            l = l * l_mask

            # Theta layer
            t, kl_term = self.l_theta(x)
            self.kl_term = kl_term

            # Context layer
            c = self.l_B(x)
            c = self.l_context(c, t)
            c = torch.mean(c, dim=2)

            # Output layer
            o = self.l_dense0(l)
            o = self.l_dense1(o)
            o = self.l_dense(o, c)
            o = self.l_out0(o)
            o = o * mask + 0.000001
            return o

    if isCONTENT:

        LEARNING_RATE = torch.tensor(Config.learning_rate, requires_grad=False)
        MAX_LENGTH =300

        X_raw_data, Y_raw_data = data_set.get_data_from_type("train")
        trainingAdmiSeqs, trainingMask, trainingLabels, trainingLengths, ltr = dp.prepare_data(X_raw_data, Y_raw_data, vocabsize=FLAGS.vocab_size, maxlen = MAX_LENGTH)
        _, MAX_LENGTH, N_VOCAB = trainingAdmiSeqs.shape

        X_valid_data, Y_valid_data = data_set.get_data_from_type("valid")
        validAdmiSeqs, validMask, validLabels, validLengths, lval  = dp.prepare_data(X_valid_data, Y_valid_data, vocabsize=FLAGS.vocab_size, maxlen = MAX_LENGTH)


        print("------CONTENT model------")
        content_model_validation = ContentModel()
        loss_fn = nn.BCELoss(reduction='sum')
        optimizer = torch.optim.Adam(content_model_validation.parameters(), lr=LEARNING_RATE)
        start_time = time.time()
        train(content_model_validation, True)
        print(f"Total time to train: {time.time()-start_time}")
        test(content_model_validation)
        process = psutil.Process(os.getpid())
        print("Total Memory Used in Training, validating and testing: {} MB".format(process.memory_info().rss / 1024)) 

    if isRNN:
        FLAGS = Config()
        data_set = Data_Loader(FLAGS)

        LEARNING_RATE = torch.tensor(Config.learning_rate, requires_grad=False)
        MAX_LENGTH =300

        X_raw_data, Y_raw_data = data_set.get_data_from_type("train")
        trainingAdmiSeqs, trainingMask, trainingLabels, trainingLengths, ltr = dp.prepare_data(X_raw_data, Y_raw_data, vocabsize=FLAGS.vocab_size, maxlen = MAX_LENGTH)
        _, MAX_LENGTH, N_VOCAB = trainingAdmiSeqs.shape

        X_valid_data, Y_valid_data = data_set.get_data_from_type("valid")
        validAdmiSeqs, validMask, validLabels, validLengths, lval  = dp.prepare_data(X_valid_data, Y_valid_data, vocabsize=FLAGS.vocab_size, maxlen = MAX_LENGTH)

        rnn_model = RNN()
        loss_fn = nn.BCELoss(reduction='sum')
        optimizer = torch.optim.SGD(rnn_model.parameters(), lr=LEARNING_RATE)

        print("------RNN model------")
        start_time = time.time()
        train(rnn_model)
        print(f"Total time to train: {time.time()-start_time}")
        test(rnn_model)

    if isLSTM:
        FLAGS = Config()
        data_set = Data_Loader(FLAGS)

        LEARNING_RATE = torch.tensor(Config.learning_rate, requires_grad=False)
        MAX_LENGTH =300

        X_raw_data, Y_raw_data = data_set.get_data_from_type("train")
        trainingAdmiSeqs, trainingMask, trainingLabels, trainingLengths, ltr = dp.prepare_data(X_raw_data, Y_raw_data, vocabsize=FLAGS.vocab_size, maxlen = MAX_LENGTH)
        _, MAX_LENGTH, N_VOCAB = trainingAdmiSeqs.shape

        X_valid_data, Y_valid_data = data_set.get_data_from_type("valid")
        validAdmiSeqs, validMask, validLabels, validLengths, lval  = dp.prepare_data(X_valid_data, Y_valid_data, vocabsize=FLAGS.vocab_size, maxlen = MAX_LENGTH)


        content_model_lstm = ContentModel_LSTM()
        loss_fn = nn.BCELoss(reduction='sum')
        optimizer = torch.optim.SGD(content_model_lstm.parameters(), lr=LEARNING_RATE)
        print("------CONTENT model with LSTM unit------")
        start_time = time.time()
        train(content_model_lstm)
        print(f"Total time to train: {time.time()-start_time}")
        test(content_model_lstm)