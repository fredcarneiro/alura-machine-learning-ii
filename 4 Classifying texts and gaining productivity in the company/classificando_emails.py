#!-*- coding: utf8 -*-

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cross_validation import cross_val_score


classificacoes = pd.read_csv("emails.csv")
textos_puros = classificacoes['emails']
textos_quebrados = textos_puros.str.lower().str.split(' ')

dicionario = set() #conjunto. Não aceita elementos repetidos

for lista in textos_quebrados:
	dicionario.update(lista)

total_de_palavras = len(dicionario)
tuplas = zip(dicionario, xrange(total_de_palavras))
tradutor = {palavra:indice for palavra, indice in tuplas} # traduz palavras para indices

def vetorizar_texto(texto, tradutor):
	vetor = [0] * len(tradutor) # representação de uma palavra vazia
	
	for palavra in texto:
		if palavra in tradutor:
			posicao = tradutor[palavra]
			vetor[posicao] += 1	


vetores_de_texto = [vetorizar_texto(texto, tradutor) for texto in textos_quebrados]

marcas = classificacoes['classificacao']

X = np.array(vetores_de_texto)
Y = np.array(marcas.tolist())

porcentagem_de_treino = 0.8

tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_do_treino

treino_dados = X[0:tamanho_do_treino]
treino_marcacoes = Y[0:tamanho_do_treino]

validacao_dados = X[tamanho_do_treino:]
validacao_marcacoes = Y[tamanho_do_treino:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
	k = 10
	scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k)
	taxa_de_acertos = np.mean(scores)

	msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
	print(msg)
	return taxa_de_acerto

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modelo_one_vc_rest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultado_one_vc_rest = fit_and_predict("OneVsRest", modelo_one_vc_rest, treino_dados, treino_marcacoes)
resultados[resultado_one_vc_rest] = modelo_one_vc_rest


from sklearn.multiclass import OneVsOneClassifier
modelo_one_vs_one = OneVsOneClassifier(LinearSVC(random_state = 0))
resultado_one_vs_one = fit_and_predict("OneVsOne", modelo_one_vs_one, treino_dados, treino_marcacoes)
resultados[resultado_one_vs_one] = modelo_one_vs_one

from sklearn.naive_bayes import MultinomialNB
modelo_mutinomialNB = MultinomialNB()
resultadoMultinomialNB = fit_and_predict("MultinomialNB", modelo_mutinomialNB, treino_dados, treino_marcacoes)
resultados[resultadoMultinomialNB] = modelo_mutinomialNB

from sklearn.ensemble import AdaBoostClassifier
modelo_adaboost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modelo_adaboost, treino_dados, treino_marcacoes)
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


vencedor.fit(treino_dados, treino_marcacoes)
resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0*total_de_acertos/total_de_elementos

msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real e: {0}".format(taxa_de_acerto)
print(msg)