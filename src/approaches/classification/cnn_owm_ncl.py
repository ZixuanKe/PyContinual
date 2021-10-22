import sys, time
import numpy as np
import torch

dtype = torch.cuda.FloatTensor  # run on GPU
import utils
from tqdm import tqdm, trange
sys.path.append("./approaches/base/")
from cnn_base import Appr as ApprBase

########################################################################################################################

class Appr(ApprBase):

    def __init__(self, model,  logger, taskcla,  args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('DIL CONTEXTUAL CNN OWM NCL')

        return



    def train(self,t,train,valid,num_train_steps,train_data,valid_data):
        best_loss = np.inf
        best_acc = 0
        best_model = utils.get_model(self.model)
        lr = self.lr
        # patience = self.lr_patience
        self.optimizer = self._get_optimizer_owm(lr)
        nepochs = self.nepochs
        test_max = 0
        # Loop epochs
        try:
            for e in range(nepochs):
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')

                self.train_epoch(t,train, iter_bar, cur_epoch=e, nepoch=nepochs)
                clock1=time.time()

                train_loss, train_acc,train_f1_macro = self.eval(t,train)

                # print('time: ',float((clock1-clock0)*30*25))

                print('| [{:d}/10], Epoch {:d}/{:d}, | Train: loss={:.3f}, acc={:2.2f}% |'.format(t + 1, e + 1,
                                                                                                 nepochs, train_loss,
                                                                                                 100 * train_acc),
                      end='')
                # # Valid
                valid_loss, valid_acc,valid_f1_macro = self.eval(t,valid)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss, 100 * valid_acc), end='')
                print()

                if valid_loss < best_loss:
                   best_loss = valid_loss
                   best_model = utils.get_model(self.model)
                   patience = self.lr_patience
                   print(' *', end='')
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= self.lr_factor
                        print(' lr={:.1e}'.format(lr), end='')
                        if lr < self.lr_min:
                            print()
                            break

                        patience = self.lr_patience
                        self.optimizer = self._get_optimizer_owm(lr)
                print()

        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model, best_model)
        return

    def train_epoch(self, t,data,iter_bar, cur_epoch=0, nepoch=0):
        self.model.train()

        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            images,targets= batch


            # Forward
            output_dict = self.model.forward(images)
            x_list=output_dict['x_list']
            h_list=output_dict['h_list']
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]

            loss = self.ce(output, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            lamda = step / len(batch)/nepoch + cur_epoch/nepoch

            alpha_array = [1.0 * 0.00001 ** lamda, 1.0 * 0.0001 ** lamda, 1.0 * 0.01 ** lamda, 1.0 * 0.1 ** lamda]

            def pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):
                x=x.detach()
                p=p.detach()


                if cnn:
                    _, _, H, W = x.shape
                    F, _, HH, WW = w.shape
                    S = stride  # stride
                    Ho = int(1 + (H - HH) / S)
                    Wo = int(1 + (W - WW) / S)
                    for i in range(Ho):
                        for j in range(Wo):
                            # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                            r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                            # r = r[:, range(r.shape[1] - 1, -1, -1)]
                            k = torch.mm(p, torch.t(r))
                            p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
                else:
                    r = x
                    k = torch.mm(p, torch.t(r))
                    p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data, torch.t(p.data))
            # Compensate embedding gradients
            for n, w in self.model.named_parameters():
                if n == 'c1.weight':
                    pro_weight(self.Pc1, x_list[0], w, alpha=alpha_array[0], stride=2)

                if n == 'c2.weight':
                    pro_weight(self.Pc2, x_list[1], w, alpha=alpha_array[0], stride=2)

                if n == 'c3.weight':
                    pro_weight(self.Pc3, x_list[2], w, alpha=alpha_array[0], stride=2)

                if n == 'fc1.weight':
                    pro_weight(self.P1,  h_list[0], w, alpha=alpha_array[1], cnn=False)

                if n == 'fc2.weight':
                    pro_weight(self.P2,  h_list[1], w, alpha=alpha_array[2], cnn=False)


            # Apply step
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        return

    def eval(self,t,data,test=None,trained_task=None):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        target_list = []
        pred_list = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                images,targets= batch
                real_b=images.size(0)
                # Forward

                # Forward
                output_dict = self.model.forward(images)
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs = output_dict['y']
                    if self.args.ent_id: #detected id
                        output_d= self.ent_id_detection(trained_task,images,t=t)
                        output = output_d['output']
                    else:
                        output = outputs[t]

                loss = self.ce(output, targets)
                _, pred = output.max(1)
                hits = (pred % 10 == targets).float()
                target_list.append(targets)
                pred_list.append(pred)
                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b
            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')

        return total_loss / total_num, total_acc / total_num,f1
