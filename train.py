import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
import tqdm
import torch.nn.functional as F


def train(n_channels=1, n_classes=2, out_pth=None, model_name='unet', folds=5, epochs=25, lr=10e-3):

    if out_pth is None:
        out_pth = os.path.join(os.getcwd(), 'model_weights')

    try:
        os.mkdir(out_pth)
    except:
        pass

    metrics = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_start = 0

    for i in range(fold_start, folds):

        fold_name = 'fold'+str(i)

        best_loss = float('inf')

        if model_name.lower() == 'unet':
            model = Unet(n_channels, n_classes)

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        best_model = model_name + '_weights.pth'

        metrics[i] = []

        hmc_loader = {x: DataLoader(dataset[i][x], batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
                      for x in ['train', 'valid']
                      }

        for epoch in range(epochs):

            print(f'Fold {i} Epoch {epoch+1}/{epochs}')
            print('-'*60)

            for phase in ['train', 'valid']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                n = 0
                sets = 0

                total_acc = 0
                total_pre = 0
                total_rec = 0
                total_f1s = 0
                total_spe = 0

                with tqdm.tqdm(total=len(hmc_loader[phase])) as pbar:

                    for img, gt_msk in hmc_loader[phase]:

                        img = img.to(device)
                        gt_msk = gt_msk.to(device)

                        with torch.set_grad_enabled(phase == 'train'):

                            pr_msk = model(img)

                            loss = criterion(pr_msk, gt_msk)

                            msk_pr = F.one_hot(F.softmax(pr_msk, dim=1).argmax(
                                dim=1), n_classes).permute(0, 3, 1, 2).float()
                            msk_gt = F.one_hot(gt_msk, n_classes).permute(
                                0, 3, 1, 2).float()

                            if phase == 'train':

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                            total_acc += batch_metric(msk_pr, msk_gt, accuracy)
                            total_pre += batch_metric(msk_pr,
                                                      msk_gt, precision)
                            total_rec += batch_metric(msk_pr, msk_gt, recall)
                            total_f1s += batch_metric(msk_pr, msk_gt, f_score)
                            total_spe += batch_metric(msk_pr,
                                                      msk_gt, specificity)

                            running_loss += loss.item() * img.size(0)
                            n += 1
                            sets += img.size(0)

                            pbar.set_postfix_str("{:.2f} ({:.2f})".format(
                                running_loss / sets, loss.item()))
                            pbar.update()

                epoch_loss = running_loss / sets
                epoch_acc = total_acc / n
                epoch_rec = total_rec / n
                epoch_spe = total_spe / n
                epoch_pre = total_pre / n
                epoch_f1s = total_f1s / n

                if phase == 'valid' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model, os.path.join(out_pth, 'best_loss.pth'))
                    print("Model Saved!")

                print(f'{phase.title()} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f} Sensitivity: {epoch_rec:.4f} Specificity: {epoch_spe:.4f} Precision: {epoch_pre:.4f} F1: {epoch_f1s:.4f}')
