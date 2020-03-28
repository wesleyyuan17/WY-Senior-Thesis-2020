import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prices = pd.read_csv('prices_aws.csv')
informed_agents_pnl = pd.read_csv('informed_agents_pnl_aws.csv')
informed_loss = pd.read_csv('informed_training_loss_aws.csv')

uninformed_agents_pnl = pd.read_csv('uninformed_agents_pnl_aws.csv')
uninformed_loss = pd.read_csv('uninformed_training_loss_aws.csv')
# loss = loss[500:]

# plot graph of price movements
fig1 = plt.figure(figsize=(50,10))
plt.plot(range(0,len(prices)), prices)
plt.title('Price Movements')
fig1.savefig("./Plots/price_movements_aws.png")

# plot portfolio value of informed agent
fig2 = plt.figure(figsize=(50,10))
plt.plot(range(0, len(informed_agents_pnl)), informed_agents_pnl)
plt.title('Informed Agent Net PnL')
fig2.savefig("./Plots/informed_agent_training_pnl_aws.png")

# plot loss
fig3 = plt.figure(figsize=(50,10))
plt.plot(range(0, len(informed_loss)), informed_loss)
plt.title('Training Loss')
fig3.savefig("./Plots/informed_training_loss_aws.png")

# plot portfolio value of uninformed agent
fig2 = plt.figure(figsize=(50,10))
plt.plot(range(0, len(uninformed_agents_pnl)), uninformed_agents_pnl)
plt.title('Uninformed Agent Net PnL')
fig2.savefig("./Plots/uninformed_agent_training_pnl_aws.png")

# plot loss
fig3 = plt.figure(figsize=(50,10))
plt.plot(range(0, len(uninformed_loss)), uninformed_loss)
plt.title('Training Loss')
fig3.savefig("./Plots/uninformed_training_loss_aws.png")

# evaluation of current agents?
