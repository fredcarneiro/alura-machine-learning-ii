import pandas as pd
from collections import Counter
from dados import fit_and_predict

data_frame = pd.read_csv('situacao_do_cliente.csv')

X_df = data_frame[['recencia', 'frequencia', 'semanas_de_inscricao']]
Y_df = data_frame['situacao']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1
porcentagem_de_validacao = 0.1

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_teste = int(porcentagem_de_teste*len(Y))
tamanho_de_validacao = int(porcentagem_de_validacao*len(Y))

fim_de_teste = tamanho_de_teste+tamanho_de_treino

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modelo_one_vc_rest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultado_one_vc_rest = fit_and_predict("OneVsRest", modelo_one_vc_rest, teste_dados, teste_marcacoes, treino_dados, treino_marcacoes)
resultados[resultado_one_vc_rest] = modelo_one_vc_rest


from sklearn.multiclass import OneVsOneClassifier
modelo_one_vs_one = OneVsOneClassifier(LinearSVC(random_state = 0))
resultado_one_vs_one = fit_and_predict("OneVsOne", modelo_one_vs_one, teste_dados, teste_marcacoes, treino_dados, treino_marcacoes)
resultados[resultado_one_vs_one] = modelo_one_vs_one


from sklearn.naive_bayes import MultinomialNB
modelo_mutinomialNB = MultinomialNB()
resultadoMultinomialNB = fit_and_predict("MultinomialNB", modelo_mutinomialNB, teste_dados, teste_marcacoes, treino_dados, treino_marcacoes)
resultados[resultadoMultinomialNB] = modelo_mutinomialNB

from sklearn.ensemble import AdaBoostClassifier
modelo_adaboost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modelo_adaboost, teste_dados, teste_marcacoes, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modelo_adaboost

print resultados
maximo = max(resultados)
vencedor = resultados(maximo)

# eficacia do algoritmo que chuta tudo 1 ou 0
# tudo o mesmo numero
acerto_base = max(Counter(validacao_marcacoes).values())

#acerto_de_um = len(Y[Y==1])
#acerto_de_zero = len(Y[Y==0])
# outro exempo X[Y==0]  Traz todas as caracteristicas dos elementos onde Y e igual a 0
# outro exempo X[Y==1]  Traz todas as caracteristicas dos elementos onde Y e igual a 1
taxa_de_acerto_base = 100.0 * acerto_base/len(validacao_marcacoes)
print("Taxa de acerto algoritmo base: %f" % taxa_de_acerto_base)

resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0*total_de_acertos/total_de_elementos

msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real e: {0}".format(taxa_de_acerto)
print(msg)