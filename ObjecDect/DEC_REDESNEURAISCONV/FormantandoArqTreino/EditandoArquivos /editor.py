#editando nome dos arquivos do dataset para treino da rede neural


import os
os.chdir("/Users/arnaldoljr/Downloads/datasets/catsvsdogs/images")

i=0

for filename in os.listdir():
    os.rename(filename,filename[:3]+str(i)+filename[4:])
    i+=1


