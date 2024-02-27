import mindspore as ms
from mindspore.train import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import nn
from model.esm1b import ProteinBertModel
from model.dictionary_promptprotein import Alphabet
from parsing import parse_train_args
from poprogress import simple_progress
from utils.conventer import PromptConverter
import lmdb
import csv
from fairseq import modules
import numpy as np
import mindspore.ops as ops
from poprogress import simple_progress
import pandas as pd

prompts = ['<seq>']

class RandomAccessDataset:
    def __init__(self, path, len):
        self.data = []
        for i in simple_progress(range(len) ,desc='converter processing'):
            a = np.load(path +"%d.npy"%(i), allow_pickle=True)
            a[0] = ms.Tensor(a[0])
            a[1] = ms.Tensor(a[1], dtype = ms.dtype.int32)
            a = list(a)
            #a.append([a[0],a[1]])
            #print(type(a))
            self.data.append(a)

    def __getitem__(self, id):
        '''overrode the getitem method to support random access'''
        return self.data[id]
    def __len__(self):
        '''specify the length of data'''
        return len(self.data)

class RandomAccessDataset1:
    def __init__(self, dictionary):
        self.data = []
        data = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
        ]
        data2=["MMSNFNMSILDEKKLTLLDKYMDGFDDKEHNIITILHYAQDIFDYLPKELQLYIARKIGIPASKVNGIVSFYSFFNENPTGKYVANVCMGTACFVKHSQDILDEFNKILKLDENGMSADKLFSINSIRCLGACGIGPVVKINDKIFGHVKKEDVAGIIKSYRDKEGL"]
        converter = PromptConverter(dictionary)
        for i in range(1):
            encoded_sequence = converter(data2[i], prompt_toks=prompts)
            # .data.append((encoded_sequences[i], self.value[i]))
            # print(encoded_sequence)
            list_data = [encoded_sequence, data2[i]]
            self.data.append(list_data)

    def __getitem__(self, id):
        '''overrode the getitem method to support random access'''
        return self.data[id]
    def __len__(self):
        '''specify the length of data'''
        return len(self.data)


def train_net():
    ms.set_context(device_target="CPU", mode=ms.PYNATIVE_MODE)

    dictionary = Alphabet.build_alphabet()
    train_data = RandomAccessDataset('./data/nptrain4/', 20930000)
    #train_data = RandomAccessDataset1(dictionary)
    train_loader = ms.dataset.GeneratorDataset(train_data,
                                               column_names=['masked_token','origin_token'],
                                               shuffle=True,
                                               num_parallel_workers=1,
                                               )

    #test_data = RandomAccessDataset1(dictionary)
    test_data = RandomAccessDataset('./data/nptest4/', 5220000)
    test_loader = ms.dataset.GeneratorDataset(test_data,
                                              column_names=['masked_token','origin_token'],
                                              shuffle=True,
                                              num_parallel_workers=1,
                                              )

    args = parse_train_args()
    model = ProteinBertModel(args, dictionary)

    loss_fn= nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    scheduler = nn.exponential_decay_lr(learning_rate=3e-4, decay_rate=0.99, total_step=6, decay_epoch=1,
                                        step_per_epoch=2)
    optimizer = nn.Adam([{"params": model.trainable_params()}], learning_rate=scheduler)

    # steps_per_epoch = train_loader.get_dataset_size()
    # config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch)
    #
    # ckpt_callback = ModelCheckpoint(prefix="esm1b", directory="./checkpoint", config=config)
    # loss_callback = LossMonitor(steps_per_epoch)
    # trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    # trainer.fit(10 , train_loader, test_loader, callbacks=[ckpt_callback, loss_callback])
    # Define forward function

    df_1 = pd.DataFrame(columns=['step', 'train Loss'])
    df_1.to_csv("./data/train_loss_3e-6.csv", index=False)
    df_1 = pd.DataFrame(columns=['step', 'train Loss'])
    df_1.to_csv("./data/train_accuracy_3e-6.csv", index=False)

    def forward_fn(data, label):
        logits = model(data, with_prompt_num=1)['logits']
        criterion = nn.CrossEntropyLoss()
        #loss = criterion(logits, label)
        logits = logits.view(-1, ops.shape(logits)[-1])
        logits = logits[1:-1]
        label = label.view(-1)
        loss = criterion(logits,label)
        return loss, logits

    # Get gradient function
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    def train_loop(model, dataset, t):
        size = dataset.get_dataset_size()
        model.set_train()
        for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
            loss = train_step(data, label)
            current_step = t * size + batch
            if batch % 5 == 0:
                loss, current = loss.asnumpy(), batch
                print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
                current_step = t * size + batch
                if current_step % 5 == 0:
                    list_loss = [current_step, loss]
                    data_loss = pd.DataFrame([list_loss])
                    data_loss.to_csv("./data/train_loss_3e-6.csv", mode='a', header=False, index=False)
            if current_step % 500 == 0:
                num_batches = dataset.get_dataset_size()
                model.set_train(False)
                total, test_loss, correct = 0, 0, 0
                for data, label in dataset.create_tuple_iterator():
                    logits = model(data, with_prompt_num=1)['logits']
                    total += len(data)
                    logits = logits.view(-1, ops.shape(logits)[-1])
                    logits = logits[1:-1]
                    label = label.view(-1)
                    test_loss += loss_fn(logits, label).asnumpy()
                    correct += (logits.argmax(1) == label).asnumpy().sum()
                test_loss /= num_batches
                correct /= total
                print(f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
                current_step = t * size +batch
                if current_step % 5 == 0:
                    list_accuracy = [current_step, (100 * correct)]
                    data_accuracy = pd.DataFrame([list_accuracy])
                    data_accuracy.to_csv("./data/train_acuuracy_3e-6.csv", mode='a', header=False, index=False)


    def test_loop(model, dataset, loss_fn, t):
        num_batches = dataset.get_dataset_size()
        model.set_train(False)
        total, test_loss, correct = 0, 0, 0
        for data, label in dataset.create_tuple_iterator():
            pred = model(data, with_prompt_num=1)['logits']
            total += len(data)
            test_loss += loss_fn(pred, label).asnumpy()
            correct += (pred.argmax(1) == label).asnumpy().sum()
        test_loss /= num_batches
        correct /= total
        print(f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        current_step = t * num_batches
        if current_step % 5 == 0:
            list_accuracy = [current_step, (100 * correct)]
            data_accuracy = pd.DataFrame([list_accuracy])
            data_accuracy.to_csv("./data/train_acuuracy4.csv", mode='a', header=False, index=False)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(model, train_loader, t)
        test_loop(model, test_loader, loss_fn, t)
        if epochs % 2 == 0:
            ms.save_checkpoint(model, "./ckpt/model_" + "%d.ckpt" % (epochs + 1))
    print("Done!")

if __name__ == '__main__':
    train_net()