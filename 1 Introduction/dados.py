import csv

def carregar_acessos():

	X = [] # dados
	Y = [] # marcacoes

	arquivo = open('dados.csv')
	leitor = csv.reader(arquivo)

	next(leitor)

	for home, busca, contato, comprou in leitor:

		# Busca e um dado categorico. Nao e uma resposta do tipo 0 ou 1.

		dados = [int(home), busca, int(contato)]
		marcacao = int(comprou)

		X.append(dados)
		Y.append(marcacao)

	return X, Y

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
	modelo.fit(treino_dados, treino_marcacoes) # treinamento

	resultado = modelo.predict(teste_dados)
	diferencas = resultado - teste_marcacoes

	acertos = [d for d in diferencas if d ==0]
	total_de_acertos = len(acertos)
	total_de_elementos = len(teste_dados)
	taxa_de_acerto = 100.0*total_de_acertos/total_de_elementos

	msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
	print(msg)
	return taxa_de_acerto