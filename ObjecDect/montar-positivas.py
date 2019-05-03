import os

largura = 50
altura = 50
pasta = 'positivas_final'
arquivo = 'positivas.lst'

for img in os.listdir(pasta):
    linha = img + ' 1 0 0 '+ str(largura) +' '+ str(altura) +'\n'

    with open(arquivo, 'a') as f:
        f.write(linha)
        print(linha)
