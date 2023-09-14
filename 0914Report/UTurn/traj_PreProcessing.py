import numpy as np
import pandas as pd

data = pd.read_csv('datasets_traj_UTurn_env21.csv')
sight = 27
traj_tx = data.loc[:, "traj_tx"].values
traj_tx_last = traj_tx[-1]
traj_ty = data.loc[:, "traj_ty"].values
traj_ty_last = traj_ty[-1]
num = data.loc[:, "num"].values

tx_spaces = []
ty_spaces = []
for i in range(20):
    tx_space = traj_tx[-1-i] - traj_tx[-2-i]
    tx_spaces.append(tx_space)

    ty_space = traj_ty[-1-i] - traj_ty[-2-i]
    ty_spaces.append(ty_space)

avg_spacex = np.sum(tx_spaces) / 20
avg_spacey = np.sum(ty_spaces) / 20

lookahead_traj_tx = []
lookahead_traj_ty = []
kx=0
ky=0
while np.abs(kx) <= sight + 1:
    kx += avg_spacex
    ky += avg_spacey
    lookahead_traj_tx.append(traj_tx_last + kx)
    lookahead_traj_ty.append(8.25)
#    lookahead_traj_ty.append(traj_ty_last + ky)
new_traj_tx = np.append(traj_tx, lookahead_traj_tx)
new_traj_ty = np.append(traj_ty, lookahead_traj_ty)


new_traj = np.column_stack((new_traj_tx, new_traj_ty))
indexes= ["traj_tx", "traj_ty"]
new_data = pd.DataFrame(data=new_traj, columns=indexes)
new_data.to_csv("datasets_traj_SLALOM_env21_1.csv")