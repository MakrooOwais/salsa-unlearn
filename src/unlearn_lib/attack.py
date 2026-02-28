import numpy as np
import torch

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn.functional import softmax
from xgboost import XGBClassifier


class Attacker:
    def __init__(self, model, train_loader, unlearn_loader, test_loader):
        self.model = model.cuda()
        self.model.eval()
        self.tr_loader = train_loader
        self.ul_loader = unlearn_loader
        self.ts_loader = test_loader
        self.attacker = self.attack_model()

    def attack_model(self):
        N_unlearn_sample = 5000
        train_data = torch.zeros([1, 10])
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.tr_loader):
                data = data.cuda()
                out = self.model(data)
                train_data = torch.cat([train_data, out.cpu()])
                if train_data.shape[0] > N_unlearn_sample:
                    break

        train_data = train_data[1:, :]
        train_data = softmax(train_data, dim=1)
        train_data = train_data.detach().numpy()

        N_unlearn_sample = 5000
        test_data = torch.zeros([1, 10])
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.ts_loader):
                data = data.cuda()
                out = self.model(data)
                test_data = torch.cat([test_data, out.cpu()])

                if test_data.shape[0] > N_unlearn_sample:
                    break

        test_data = test_data[1:, :]
        test_data = softmax(test_data, dim=1)
        test_data = test_data.detach().numpy()

        att_y = np.hstack((np.ones(train_data.shape[0]), np.zeros(test_data.shape[0])))
        att_y = att_y.astype(np.int16)

        att_X = np.vstack((train_data, test_data))
        att_X.sort(axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            att_X, att_y, test_size=0.01, shuffle=True
        )

        attacker = XGBClassifier(
            n_estimators=100,
            n_jobs=-1,
            objective="binary:logistic",
            booster="gbtree",
        )

        attacker.fit(X_train, y_train)

        pred_Y = attacker.predict(X_train)
        acc = accuracy_score(y_train, pred_Y)

        pred_Y_test = attacker.predict(X_test)
        acc_test = accuracy_score(y_test, pred_Y_test)

        # print("MIA attacker training accuracy = {:.4f}".format(acc))
        # print("MIA attacker testing accuracy = {:.4f}".format(acc_test))

        return attacker
    
    def attack(self):
        unlearn_X = torch.zeros([1, 10])
        unlearn_X = unlearn_X
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.ul_loader):
                data = data.cuda()
                out = self.model(data)
                unlearn_X = torch.cat([unlearn_X, out.cpu()])

        unlearn_X = unlearn_X[1:, :]
        unlearn_X = softmax(unlearn_X, dim=1)
        unlearn_X = unlearn_X.detach().numpy()

        unlearn_y = np.ones(unlearn_X.shape[0])
        unlearn_y = unlearn_y.astype(np.int16)

        N_unlearn_sample = unlearn_X.shape[0]

        test_X = torch.zeros([1, 10])
        i = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.ts_loader):
                if i < 5000:
                    i += data.shape[0]
                else:
                    data = data.cuda()
                    out = self.model(data)
                    test_X = torch.cat([test_X, out.cpu()])

                    if test_X.shape[0] > N_unlearn_sample:
                        break

        test_X = test_X[1:, :]
        test_X = softmax(test_X, dim=1)
        test_X = test_X.detach().numpy()

        test_y = np.zeros(test_X.shape[0])
        test_y = test_y.astype(np.int16)

        y = np.hstack((unlearn_y, test_y))
        y = y.astype(np.int16)

        X = np.vstack((unlearn_X, test_X))
        X.sort(axis=1)

        pred_Y = self.attacker.predict(X)
        acc = accuracy_score(y, pred_Y)

        print("MIA Attacker accuracy = {:.4f}".format(acc))
        return acc
