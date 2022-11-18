import torch
import torch.optim as optim
import numpy as np
from models import pre_trained_nets
from models import fairness_regularizer
import logging


class BinaryAttributeClassifier:

    def __init__(self, model_name: str,
                 tuning: str = 'last_layer',
                 two_headed: bool = False,
                 second_head_weight: float = 0.5,
                 optimizer_settings: dict = None,
                 model_path: str = None,
                 only_warmstart: bool = False,
                 print_freq: int = 100,
                 writer: object = None,
                 device: str = 'cuda',
                 pretrained: bool = True,
                 dtype: type = torch.float32):

        if model_path is not None:
            A = torch.load(model_path, map_location=device)
            model_name = A['model_name']

        self.model_name = model_name
        self.number_of_targets = 1 + two_headed
        self.two_headed = two_headed
        self.second_head_weight = second_head_weight
        self.model = pre_trained_nets.get_pre_trained_model(self.model_name,
                                                            n_classes=self.number_of_targets,
                                                            pretrained=pretrained,
                                                            require_all_grads=tuning == 'full_pass')

        if optimizer_settings is not None:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=optimizer_settings['learning_rate'],
                                        weight_decay=0)
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=1e-4,
                                        weight_decay=0)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=15, threshold=0.0001, verbose=True)

        self.device = device
        self.dtype = dtype
        self.writer = writer

        self.epoch = 0
        self.best_eval_measure = 0.0
        self.print_freq = print_freq

        if model_path is not None:
            self.model.load_state_dict(A['model'])
            if self.device == torch.device('cuda'):
                self.model.cuda()
            if not only_warmstart:
                self.optimizer.load_state_dict(A['optim'])
                self.epoch = A['epoch']
                self.best_eval_measure = A['best_average_precision']

    def forward(self, x):
        out = self.model(x)
        return out

    def save_model(self, path):
        torch.save({'model_name': self.model_name, 'model': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(), 'epoch': self.epoch,
                    'best_average_precision': self.best_eval_measure}, path)

    def train(self, loader):
        """Train the model for one epoch"""

        self.model.train()
        running_loss = 0.0
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        criterion = torch.nn.BCEWithLogitsLoss()  # Combines Sigmoid and BCE in one layer
        if self.two_headed:
            second_head_criterion = fairness_regularizer.get_second_head_criterion(second_head_criterion='MSE')
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(device=self.device, dtype=self.dtype), targets.to(device=self.device,
                                                                                          dtype=self.dtype)
            self.optimizer.zero_grad()
            outputs = self.forward(images)
            loss = criterion(outputs[:, 0].squeeze(), targets[:, 0])
            if self.two_headed:
                loss = (1 - self.second_head_weight) * loss + \
                       self.second_head_weight * second_head_criterion(outputs[:, 1].squeeze(), targets[:, 1])

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if self.print_freq and (i % self.print_freq == (self.print_freq-1)):
                logging.info('Training epoch {}: [{}|{}], average loss:{}'.format(
                    self.epoch, i + 1, len(loader), running_loss / self.print_freq))
                if self.writer:
                    self.writer.add_scalar('training_loss', running_loss / self.print_freq, self.epoch * len(loader) + i)
                running_loss = 0.0
        self.epoch += 1

    def predict(self, loader, threshold=0):

        return np.where(self.predict_proba(loader) > threshold, 1, 0)

    def predict_proba(self, loader):
        if self.device == torch.device('cuda'):
            self.model.cuda()
        self.model.eval()
        scores_all = []
        with torch.no_grad():
            for (x, y) in loader:
                x = x.to(device=self.device, dtype=self.dtype)

                scores = self.model(x)
                scores = torch.sigmoid(scores).squeeze()

                scores_all.append(scores.detach().cpu().numpy())
            y_score = np.concatenate(scores_all)

        return y_score

    def get_true_label_and_score_prediction(self, loader):

        if self.device == torch.device('cuda'):
            self.model.cuda()
        self.model.eval()
        y_true_all = []
        scores_all = []
        average_loss = 0
        lossbce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        if self.two_headed:
            second_head_criterion = fairness_regularizer.get_second_head_criterion(second_head_criterion='MSE')
        with torch.no_grad():
            for (x, y) in loader:
                x = x.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)

                outputs = self.model(x)
                loss = lossbce(outputs[:, 0].squeeze(), y[:, 0]).item()
                if self.two_headed:
                    loss = (1-self.second_head_weight) * loss + self.second_head_weight * second_head_criterion(outputs[:, 1].squeeze(), y[:, 1])

                average_loss += loss
                scores = torch.sigmoid(outputs).squeeze()

                y_true_all.append(y.detach().cpu().numpy())
                scores_all.append(scores.detach().cpu().numpy())
            y_true_all = np.concatenate(y_true_all)
            y_score_all = np.concatenate(scores_all)

        return y_true_all, y_score_all, average_loss / len(loader)

    # This functions only works for two_headed models
    def get_weighted_average_aware_prediction(self, loader, second_head_weight=0.5):

        if not self.two_headed:
            print('This function can only be used with second head')

        if self.device == torch.device('cuda'):
            self.model.cuda()
        self.model.eval()
        y_true_all = []
        scores_all = []
        with torch.no_grad():
            for (x, y) in loader:
                x = x.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)

                outputs = self.model(x)
                scores = (1 - second_head_weight) * outputs[:, 0] + second_head_weight * outputs[:, 1]

                y_true_all.append(y.detach().cpu().numpy())
                scores_all.append(scores.detach().cpu().numpy())
            y_true_all = np.concatenate(y_true_all)
            y_score_all = np.concatenate(scores_all)

        return y_true_all, y_score_all


class FairBinaryAttributeClassifier(BinaryAttributeClassifier):

    def __init__(self, model_name: str,
                    tuning: str = 'last_layer',
                    fairness_notion: str = 'DDP',
                    fairness_parameter: int = 100,
                    optimizer_settings: dict = None,
                    model_path: str = None,
                    only_warmstart: bool = False,
                    print_freq: int = 100,
                    writer: object = None,
                    device: str = 'cuda',
                    dtype: type = torch.float32):

        super(FairBinaryAttributeClassifier, self).__init__(model_name=model_name,
                                                            device=device,
                                                            optimizer_settings=optimizer_settings,
                                                            two_headed=False,
                                                            tuning=tuning,
                                                            dtype=dtype,
                                                            model_path=model_path,
                                                            writer=writer,
                                                            print_freq=print_freq,
                                                            only_warmstart=only_warmstart)

        self.fairness_notion = fairness_notion
        self.fairness_parameter = fairness_parameter

    def train(self, loader):
        """Train the model for one epoch"""

        self.model.train()
        running_loss = 0.0
        running_fairness_loss = 0.0
        criterion = torch.nn.BCEWithLogitsLoss()  # Combines Sigmoid and BCE in one layer
        fairness_criterion = fairness_regularizer.get_fairness_criterion(self.fairness_notion)

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(device=self.device, dtype=self.dtype), targets.to(device=self.device,
                                                                                          dtype=self.dtype)
            self.optimizer.zero_grad()
            outputs = self.forward(images)
            loss = criterion(outputs[:, 0].squeeze(), targets[:, 0])
            fairness_loss = fairness_criterion(images, outputs[:, 0], targets[:, 1], targets[:, 0])
            loss = loss + self.fairness_parameter * fairness_loss

            loss.backward()
            self.optimizer.step()


            running_loss += loss.item()
            running_fairness_loss += fairness_loss.item()
            if self.print_freq and (i % self.print_freq == (self.print_freq - 1)):
                logging.info('Training epoch {}: [{}|{}], average loss:{}, average fair loss:{}'.format(
                    self.epoch, i + 1, len(loader), running_loss / self.print_freq, running_fairness_loss / self.print_freq))
                if self.writer:
                    self.writer.add_scalar('training_loss', running_loss / self.print_freq,
                                           self.epoch * len(loader) + i)
                    self.writer.add_scalar('fairness_loss', running_fairness_loss / self.print_freq,
                                           self.epoch * len(loader) + i)
                running_loss = 0.0
                running_fairness_loss = 0.0
        self.epoch += 1