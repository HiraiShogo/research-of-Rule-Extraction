'''
Created on 2019/06/21

@author: ail
'''

from dnn_learn import Model  # @UnresolvedImport
import os
import subprocess


if __name__ == '__main__':
    #==========================================================
    #変数設定

    gpu_device = 0
    epoch = 30
    batch_size = 100
    frequency = -1
    n_in = 784 #MNIST = 784
    n_units = 100
    n_out = 10
    #===========================================================

    print("Input folder Name")
    name = input(' name > ')
    path = "result\\"+name
    while(os.path.exists(path)):
        print()
        print("This path has already used!")
        print("Input again.")
        name = input(' name > ')
        path = "result\\"+name
    os.makedirs(path)

    model = Model()
    model.train_Network(n_in, n_units, n_out, gpu_device, epoch, batch_size, frequency, path)
    model.save_Network(path)



    subprocess.Popen(['explorer',path])
    #model.load_Network()
    #out = model.predict()
    #print(out["Pool"])

