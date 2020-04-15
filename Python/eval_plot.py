import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from constants import *

sns.set() # formatting

case = sys.argv[1]

if case == 'train': # look at training data
	prices = pd.read_csv('prices_aws.csv')
	informed_agents_pnl = pd.read_csv('informed_agents_pnl_aws.csv')
	informed_loss = pd.read_csv('informed_training_loss_aws.csv')

	uninformed_agents_pnl = pd.read_csv('uninformed_agents_pnl_aws.csv')
	uninformed_loss = pd.read_csv('uninformed_training_loss_aws.csv')

	# plot graph of price movements
	fig1 = plt.figure(figsize=(50,10))
	plt.plot(range(len(prices)), prices)
	plt.title('Price Movements', fontsize=35)
	plt.xticks(np.arange(0, DEBUG_ROUNDS, 5000))
	fig1.axes[0].grid(which='minor', color='2', linestyle=':', linewidth=2)
	fig1.savefig("./Plots/price_movements_aws.png")

	# plot portfolio value of informed agent
	fig2 = plt.figure(figsize=(50,10))
	plt.plot(range(len(informed_agents_pnl)), informed_agents_pnl)
	plt.title('Informed Agent Net PnL', fontsize=35)
	plt.xticks(np.arange(0, DEBUG_ROUNDS, 5000))
	fig2.axes[0].grid(which='minor', color='2', linestyle=':', linewidth=2)
	fig2.savefig("./Plots/informed_agent_training_pnl_aws.png")

	# plot loss
	fig3 = plt.figure(figsize=(50,10))
	plt.plot(range(len(informed_loss)), informed_loss)
	plt.title('Training Loss', fontsize=35)
	plt.xticks(np.arange(0, DEBUG_ROUNDS, 5000))
	fig3.axes[0].grid(which='minor', color='2', linestyle=':', linewidth=2)
	fig3.savefig("./Plots/informed_training_loss_aws.png")

	# plot portfolio value of uninformed agent
	fig4 = plt.figure(figsize=(50,10))
	plt.plot(range(len(uninformed_agents_pnl)), uninformed_agents_pnl)
	plt.title('Uninformed Agent Net PnL', fontsize=35)
	plt.xticks(np.arange(0, DEBUG_ROUNDS, 5000))
	fig4.axes[0].grid(which='minor', color='2', linestyle=':', linewidth=2)
	fig4.savefig("./Plots/uninformed_agent_training_pnl_aws.png")

	# plot loss
	fig5 = plt.figure(figsize=(50,10))
	plt.plot(range(len(uninformed_loss)), uninformed_loss)
	plt.title('Training Loss', fontsize=35)
	plt.xticks(np.arange(0, DEBUG_ROUNDS, 5000))
	fig5.axes[0].grid(which='minor', color='2', linestyle=':', linewidth=2)
	fig5.savefig("./Plots/uninformed_training_loss_aws.png")

elif case == 'eval': # evaluation of current agents
	prices = pd.read_csv('prices_informed_eval.csv')
	informed_agents_pnl_base = pd.read_csv('informed_agents_pnl_eval.csv')
	informed_agents_pnl = pd.read_csv('informed_agents_pnl_aws.csv')
	uninformed_agents_pnl = pd.read_csv('uninformed_agents_pnl_aws.csv')
	pnl_sum = informed_agents_pnl + uninformed_agents_pnl

	# plot graph of price movements
	fig1 = plt.figure(figsize=(50,10))
	plt.plot(range(len(prices)), prices)
	plt.title('Price Movements', fontsize=35)
	plt.xticks(np.arange(0, DEBUG_ROUNDS, 5000))
	fig1.axes[0].grid(which='minor', color='2', linestyle=':', linewidth=2)
	fig1.savefig("./Plots/price_movements_eval.png")

	# plot portfolio value of informed agent
	fig2 = plt.figure(figsize=(50,10))
	plt.plot(range(len(informed_agents_pnl_base)), informed_agents_pnl_base, label='Informed Agent P&L - Base')
	plt.plot(range(len(informed_agents_pnl)), informed_agents_pnl, label='Informed Agent P&L - Adversarial')
	plt.plot(range(len(uninformed_agents_pnl)), uninformed_agents_pnl, label='Uninformed Agent P&L')
	plt.title('Trading Agent Net P&L', fontsize=35)
	plt.legend(fontsize=25)
	plt.xticks(np.arange(0, DEBUG_ROUNDS, 5000))
	fig2.axes[0].grid(which='minor', color='2', linestyle=':', linewidth=2)
	fig2.savefig("./Plots/training_agent_eval_pnl.png")
