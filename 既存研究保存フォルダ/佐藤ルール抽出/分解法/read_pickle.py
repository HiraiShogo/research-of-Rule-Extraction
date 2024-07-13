import pickle

#shutil.rmtree("result")#resultフォルダの削除(resultフォルダの上書き用)
#os.mkdir("result")#resultフォルダの作成

num = 5
p = "CNN_M3-2\\params"

with open(p+".pkl", mode="rb") as f:  # 読み込むファイル名
    params = pickle.load(f)

print(params.items())
weights = params['W' + str(num)]
biases = params['b' + str(num)]

print(weights.shape)

for i in range(0, 10):
    fw = open(str("read_result\\inputs[" + str(i) + "].txt"), "w")
    for j in range(0,10):
        print(weights[j,i])
        fw.write(str(weights[j,i]) +"\n")
    fw.write(str(biases[i]))
    fw.close()



#f1 = open(str("pool[" + str(i) + "][" + str(j) + "].txt"), "w")
#        f1.write(str(pool[i][j]))
 #       f1.close()