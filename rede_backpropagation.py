import os
from math import exp, sqrt
from random import uniform, shuffle, seed

#Aluna Giovanna Carreira Marinho.

def Menu():
	print('\n--------------------------------------------------------------')
	print('------ Rede Backpropagation - Giovanna Carreira Marinho ------')
	print('--------------------------------------------------------------')
	print('      Escolha a operação:')
	print('       (1) Treinamento')
	print('       (2) Teste')
	print('       (0) SAIR')	
	return(int(input('      Operação: ')))

def leituraArquivo(nomeArquivo, embaralha = True):
	arquivo = open(nomeArquivo, "r")
	entradas = []
	classes = []

	next(arquivo) #Pulando a primeira linha
	for linha in arquivo:
		linhaSemVirgula = linha.split(",")
		linhaInteira = [int(string) for string in linhaSemVirgula]

		if linhaInteira[-1] not in classes:
			classes.append(linhaInteira[-1])

		entradas.append(linhaInteira)

	arquivo.close()

	quantidadeClasses = len(classes)
	quantidadeNeuroniosCamadaEntrada = len(linhaInteira) - 1
	quantidadeNeuroniosCamadaSaida = len(classes)

	if embaralha == True: 
		shuffle(entradas) #Embaralhando as entradas para distribuir melhor as classes

	return entradas, quantidadeClasses, quantidadeNeuroniosCamadaEntrada, quantidadeNeuroniosCamadaSaida

def mediaGeometrica(x, y):
	return int(sqrt(x * y))

def logistica(x):
    return 1 / (1 + exp(-x))

def tangenteHiperbolica(x):
    return (1 - exp(-2 * x)) / (1 + exp(-2 * x))

def dLogisticaDx(x):
    return exp(-x) / ((1 + exp(-x))*(1 + exp(-x)))

def dTangenteHiperbolicaDx(x):
    return 1 - (tangenteHiperbolica(x)*tangenteHiperbolica(x))


class RedeBackpropagation():
	def __init__(self, taxaAprendizado, quantidadeClasses, quantidadeNeuroniosCamadaEntrada, quantidadeNeuroniosCamadaOculta, quantidadeNeuroniosCamadaSaida):
		self.taxaAprendizado = taxaAprendizado
		self.quantidadeNeuroniosCamadaEntrada = quantidadeNeuroniosCamadaEntrada
		self.quantidadeNeuroniosCamadaOculta = quantidadeNeuroniosCamadaOculta
		self.quantidadeNeuroniosCamadaSaida = quantidadeNeuroniosCamadaSaida
		self.matrizConfusao = [[0 for i in range(quantidadeClasses)] for j in range(quantidadeClasses)]
		#Inicializando os pesos da camada oculta com valores aleatórios entre -0.1 e 0.1 com 5 casas decimais
		self.pesosCamadaOculta = [[round(uniform(-0.1, 0.1), 5) for i in range(quantidadeNeuroniosCamadaEntrada)] for j in range(quantidadeNeuroniosCamadaOculta)]
		#Inicializando os pesos da camada de saída com valores aleatórios entre -0.1 e 0.1 com 5 casas decimais
		self.pesosCamadaSaida = [[round(uniform(-0.1, 0.1), 5) for i in range(quantidadeNeuroniosCamadaOculta)] for j in range(quantidadeNeuroniosCamadaSaida)]
		self.netCamadaOculta = [0 for i in range(quantidadeNeuroniosCamadaOculta)]
		self.netCamadaSaida = [0 for i in range(quantidadeNeuroniosCamadaSaida)]
		self.saidaCamadaOculta = [0 for i in range(quantidadeNeuroniosCamadaOculta)]
		self.saidaCamadaSaida = [0 for i in range(quantidadeNeuroniosCamadaSaida)]
		self.errosCamadaOculta = [0 for i in range(quantidadeNeuroniosCamadaOculta)]
		self.errosCamadaSaida = [0 for i in range(quantidadeNeuroniosCamadaSaida)]

	def __str__(self):
		string = ''
		string += '\n' + '        Pesos da camada oculta:'
		for i in range(len(self.pesosCamadaOculta)):
			string += '\n' + '        ' + str(self.pesosCamadaOculta[i])

		string += '\n\n' + '        Pesos da camada de saída:'
		for i in range(len(self.pesosCamadaSaida)):
			string += '\n' + '        ' + str(self.pesosCamadaSaida[i])

		string += '\n\n' + '        Erros da camada oculta:' + str(self.errosCamadaOculta)
		string += '\n\n' + '        Erros da camada de saída:' + str(self.errosCamadaSaida)
		return string

	def calculaNetsCamadaOculta(self, entradas):	#Calcula os nets da camada oculta
		for j in range(0, self.quantidadeNeuroniosCamadaOculta):
			net = 0
			for i in range(0, self.quantidadeNeuroniosCamadaEntrada):
				net += self.pesosCamadaOculta[j][i] * entradas[i]
			self.netCamadaOculta[j] = net

	def aplicaFuncaoTransferenciaCamadaOculta(self, funcao): #Aplica função de transferencia na camada oculta
		if funcao == 'Logistica':
			for j in range(0, self.quantidadeNeuroniosCamadaOculta):
				self.saidaCamadaOculta[j] = logistica(self.netCamadaOculta[j])
		else: #funcao == 'TangenteHiperbolica'
			for j in range(0, self.quantidadeNeuroniosCamadaOculta):
				self.saidaCamadaOculta[j] = tangenteHiperbolica(self.netCamadaOculta[j])

	def calculaNetsCamadaSaida(self, entradas):	    #Calcula os nets da camada de saída
		for j in range(0, self.quantidadeNeuroniosCamadaSaida):
			net = 0
			for i in range(0, self.quantidadeNeuroniosCamadaOculta):
				net += self.pesosCamadaSaida[j][i] * self.saidaCamadaOculta[i]
			self.netCamadaSaida[j] = net

	def aplicaFuncaoTransferenciaCamadaSaida(self, funcao): #Aplica função de transferencia na camada de saída
		if funcao == 'Logistica':
			for j in range(0, self.quantidadeNeuroniosCamadaSaida):
				self.saidaCamadaSaida[j] = logistica(self.netCamadaSaida[j])
		else: #funcao == 'TangenteHiperbolica'
			for j in range(0, self.quantidadeNeuroniosCamadaSaida):
				self.saidaCamadaSaida[j] = tangenteHiperbolica(self.netCamadaSaida[j])

	def calculaErrosCamadaSaida(self, funcao, entradas): #Calcula os erros da camada de saída
		desejado = entradas[-1] #O valor desejado é o ultimo do vetor de entradas
		
		if funcao == 'Logistica':
			desejados = [0 for i in range(0, self.quantidadeNeuroniosCamadaSaida)]
			desejados[desejado-1] = 1

			for j in range(0, self.quantidadeNeuroniosCamadaSaida):
				self.errosCamadaSaida[j] = (desejados[j] - self.saidaCamadaSaida[j])*dLogisticaDx(self.netCamadaSaida[j])
		else: #funcao == 'TangenteHiperbolica'
			desejados = [-1 for i in range(0, self.quantidadeNeuroniosCamadaSaida)]
			desejados[desejado-1] = 1

			for j in range(0, self.quantidadeNeuroniosCamadaSaida):
				self.errosCamadaSaida[j] = (desejados[j] - self.saidaCamadaSaida[j])*dTangenteHiperbolicaDx(self.netCamadaSaida[j])

	def calculaErrosCamadaOculta(self, funcao): #Calcula os erros da camada oculta
		if funcao == 'Logistica':
			for j in range(0, self.quantidadeNeuroniosCamadaOculta):
				somatorio = 0
				for i in range(0, self.quantidadeNeuroniosCamadaSaida):
					somatorio += self.errosCamadaSaida[i]*self.pesosCamadaSaida[i][j]
				self.errosCamadaOculta[j] = dLogisticaDx(self.netCamadaOculta[j])*somatorio
		else: #funcao == 'TangenteHiperbolica'
			for j in range(0, self.quantidadeNeuroniosCamadaOculta):
				somatorio = 0
				for i in range(0, self.quantidadeNeuroniosCamadaSaida):
					somatorio += self.errosCamadaSaida[i]*self.pesosCamadaSaida[i][j]
				self.errosCamadaOculta[j] = dTangenteHiperbolicaDx(self.netCamadaOculta[j])*somatorio

	def atualizaPesosCamadaSaida(self): #Atualiza  os pesos da camada de saída
		for i in range(0, self.quantidadeNeuroniosCamadaSaida):
			for j in range(0, self.quantidadeNeuroniosCamadaOculta):
				self.pesosCamadaSaida[i][j] += self.taxaAprendizado * self.errosCamadaSaida[i] * self.saidaCamadaOculta[j]

	def atualizaPesosCamadaOculta(self, entradas): #Atualiza  os pesos da camada oculta
		for i in range(0, self.quantidadeNeuroniosCamadaOculta):
			for j in range(0, self.quantidadeNeuroniosCamadaEntrada):
				self.pesosCamadaOculta[i][j] += self.taxaAprendizado * self.errosCamadaOculta[i] * entradas[j]

	def calculaErroRede(self): #Calcula o erro da rede
		somatorio = 0
		for k in range(0, self.quantidadeNeuroniosCamadaSaida):
			somatorio += (self.errosCamadaSaida[k] * self.errosCamadaSaida[k])
		return somatorio/2

	def treino(self, entradas, funcao): #Realiza os 9 passos necessários para fazer 1 treinamento sobre as amostras, retornando o erro
		for i in range(len(entradas)):
			self.calculaNetsCamadaOculta(entradas[i])
			self.aplicaFuncaoTransferenciaCamadaOculta(funcao)
			self.calculaNetsCamadaSaida(entradas[i])
			self.aplicaFuncaoTransferenciaCamadaSaida(funcao)
			self.calculaErrosCamadaSaida(funcao, entradas[i])
			self.calculaErrosCamadaOculta(funcao)
			self.atualizaPesosCamadaSaida()
			self.atualizaPesosCamadaOculta(entradas[i])
		return self.calculaErroRede()

	def treinamento(self, tipo, limiar, entradas, funcao): #Realiza o treinamento até que a condição de parada seja satisfeita 
		iteracao = 0
		print('          -> Limiar: ' + str(limiar))
		if tipo == 'Erro':
			erro = limiar #apenas para inicializar a variável e entrar no loop

			while limiar <= erro: #vai parar quando o erro da rede for menor que o limiar
				erro = self.treino(entradas, funcao)
				iteracao += 1
			
		else: #tipo == 'Iteração'
			while iteracao < limiar:
				erro = self.treino(entradas, funcao)
				iteracao += 1
		print('          -> Erro final: ' + str(erro))
		print('          -> Iterações necessárioas: ' + str(iteracao))
	
	def atualizaMatrizConfusao(self, entradas, funcao): #Alterando os valores obtidos de acordo com a função utilizada e atualizar a matriz de confusão
		desejado = entradas[-1] #O valor desejado é o ultimo do vetor de entradas

		maior = 0
		for i in range(1, quantidadeNeuroniosCamadaSaida): #Pegando o primeiro maior
			if self.saidaCamadaSaida[i] > self.saidaCamadaSaida[maior]:
				maior = i

		self.matrizConfusao[maior][desejado-1] += 1 #linha: obtido (primeiro maior), coluna: desejado

	def apresentaMatrizConfusao(self): #Apresenta a matriz de confusao
		print('\n        Matriz de confusão: (Obtido x Desejado)')
		for i in range(0, len(self.matrizConfusao)):
			print('        ' + str(self.matrizConfusao[i]))

	def teste(self, entradas, funcao): #Realiza os passos para teste sobre as amostras, apresentando a matriz de confusão
		for i in range(len(entradas)):
			self.calculaNetsCamadaOculta(entradas[i])
			self.aplicaFuncaoTransferenciaCamadaOculta(funcao)
			self.calculaNetsCamadaSaida(entradas[i])
			self.aplicaFuncaoTransferenciaCamadaSaida(funcao)
			self.atualizaMatrizConfusao(entradas[i], funcao)
		self.apresentaMatrizConfusao()


if __name__ == '__main__':
	os.system('cls')

	operacao = 1

	while(operacao != 0):
		operacao = Menu()
		os.system('cls')

		if operacao == 1:
			seed()
			print('\n        TREINAMENTO')
			nomeArquivoTreinamento = input('\n        Nome do arquivo de treinamento: ')
			
			entradas, quantidadeClasses, quantidadeNeuroniosCamadaEntrada, quantidadeNeuroniosCamadaSaida = leituraArquivo(nomeArquivoTreinamento, embaralha=True)
			
			quantidadeNeuroniosCamadaOculta = mediaGeometrica(quantidadeNeuroniosCamadaEntrada, quantidadeNeuroniosCamadaSaida)

			print('          -> Neurônios na camada de entrada: ' + str(quantidadeNeuroniosCamadaEntrada))
			print('          -> Neurônios na camada oculta: ' + str(quantidadeNeuroniosCamadaOculta))
			print('          -> Neurônios na camada de saída: ' + str(quantidadeNeuroniosCamadaSaida))

			resposta = ''
			while resposta != 'S' and resposta != 'N':
				resposta = input('\n        Alterar quantidade de neurônios na camada oculta? [S/N]: ').upper()

			if resposta == 'S':
				quantidadeNeuroniosCamadaOculta = int(input('          Quantidade: '))

			print('\n        Escolha a função de transferência:')
			print('          (1) Logística')
			print('          (2) Tangente Hiperbólica')
			resposta = ''
			while resposta != '1' and resposta != '2':
				resposta = input('          Função: [1/2]: ')
			if resposta == '1':
				funcao = 'Logistica'
			elif resposta == '2':
				funcao = 'TangenteHiperbolica'

			print('\n        Escolha a condição de parada:')
			print('          (1) Erro máximo')
			print('          (2) Número de iterações')
			resposta = ''
			while resposta != '1' and resposta != '2':
				resposta = input('          Condição: [1/2]: ')
			if resposta == '1':
				limiar = float(input('            Erro: '))
				tipo = 'Erro'
			elif resposta == '2':
				limiar = int(input('            Iterações: '))
				tipo = 'Iteração'

			taxaAprendizado = float(input('\n        Taxa de aprendizado: '))

			rede = RedeBackpropagation(taxaAprendizado, quantidadeClasses, quantidadeNeuroniosCamadaEntrada, quantidadeNeuroniosCamadaOculta, quantidadeNeuroniosCamadaSaida)
			print('\n        REDE NEURAL BACKPROPAGATION CRIADA COM SUCESSO!')
			#print(rede)

			print('\n        INÍCIO DO TREINAMENTO... (pode levar algum tempo)')
			rede.treinamento(tipo, limiar, entradas, funcao)
			print('\n        FIM DO TREINAMENTO!')
			print(rede)
			print('\n        REDE NEURAL BACKPROPAGATION TREINADA COM SUCESSO!\n')
			

		elif operacao == 2:
			print('\n        TESTE')
			nomeArquivoTeste = input('\n        Nome do arquivo de teste: ')
			
			entradas, _, _, _ = leituraArquivo(nomeArquivoTeste, embaralha=False)
			
			print('\n        INÍCIO DO TESTE...')
			rede.teste(entradas, funcao)
			print('\n        FIM DO TESTE!')

		elif operacao == 0:
			print('\n        VOCÊ ESCOLHEU A OPÇÃO SAIR!')
			
		else:
			print('\n          OPERAÇÃO INVÁLIDA')
			

		os.system('pause')
		os.system('cls')