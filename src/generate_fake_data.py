import os
import argparse
import numpy as np
import pickle
import pandas as pd

from simulation.Simulate import Simulate

file_source = os.path.dirname(__file__)
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="simulation-hawkes", help="Model name")
parser.add_argument("--seed", type=int, default=2, help="Random seed")
parser.add_argument("--n_sims", type=int, default=10, help="Number of simulations")
parser.add_argument("--T", type=int, default=100, help="Time horizon")
parser.add_argument("--inputs_path", type=str, default=os.path.join(file_source, 'data', 'inputs'), help="Inputs path")
parser.add_argument("--outputs_path", type=str, default=os.path.join(file_source, 'data', 'outputs'), help="Outputs path")

if __name__ == '__main__':

    args = parser.parse_args()

    model_name = args.model_name
    seed = args.seed
    n_sims = args.n_sims
    T = args.T
    cols = [
        "lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
        "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid"
        ]
    n_cols = len(cols)

    np.random.seed(seed)
    mat = np.zeros((n_cols,n_cols))
    for i in range(n_cols):
        mat[i][i] = .66
    for i in range(n_cols):
        for j in range(n_cols):
            if i == j: continue
            mat[i][j] = np.random.choice([1,-1])*mat[i][i]*np.exp(-.75*np.abs(j-i))

    # create fake TOD params - 1.0 for now but can be changed to whatever you wish
    faketod = {}
    for k in cols:
        faketod[k] = {}
        for k1 in np.arange(13):
            faketod[k][k1] = 1.0

    # check if dir exists
    if not os.path.exists(os.path.join(args.inputs_path, model_name)):
        os.makedirs(os.path.join(args.inputs_path, model_name))

    # save the fake data
    with open(os.path.join(args.inputs_path, model_name, "fakeData_Params_sod_eod_dictTOD_constt"), "wb") as f:
        pickle.dump(faketod, f)

    # Create fake power law kernel params from the above norm matrix
    paramsFake = {}
    for i in range(n_cols):
        paramsFake[cols[i]] = 0.1*np.random.choice([0.3,0.4,0.5,0.6,0.7])
        for j in range(n_cols):
            maxTOD = np.max(list(faketod[cols[j]].values()))
            beta = np.random.choice([1.5,1.6,1.7,1.8,1.9])
            gamma = (1+np.random.rand())*5e3
            alpha = np.abs(mat[i][j])*gamma*(beta-1)/maxTOD
            paramsFake[cols[i]+"->"+cols[j]] = (np.sign(mat[i][j]), np.array([alpha, beta, gamma]))

    mat = np.zeros((n_cols,n_cols))
    for i in range(len(cols)):
        for j in range(len(cols)):
            kernelParams = paramsFake.get(cols[j]+"->"+cols[i], None)
            if kernelParams is None: continue
            mat[i][j] = kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2])

    with open(os.path.join(args.inputs_path, model_name, "fake_ParamsInferredWCutoff_sod_eod_true"), "wb") as f:
        pickle.dump(paramsFake, f)

    paramsPath = os.path.join(args.inputs_path, model_name, "fake_ParamsInferredWCutoff_sod_eod_true")
    todPath = os.path.join(args.inputs_path, model_name, "fakeData_Params_sod_eod_dictTOD_constt")
    simulate = Simulate()

    # check if dir exists
    if not os.path.exists(os.path.join(args.outputs_path, model_name)):
        os.makedirs(os.path.join(args.outputs_path, model_name))

    #### WARNING: THIS PIECE OF CODE TAKES A LONG TIME ####
    for i in range(n_sims):
        Ts, lob, lobL3 = simulate.run(T=T, # 6.5*3600
                                    paramsPath=paramsPath, 
                                    todPath=todPath, 
                                    beta=1., 
                                    avgSpread=.01, 
                                    spread0=5, 
                                    price0=45,
                                    verbose=True)
        
        if len(pd.DataFrame(Ts[1:])[0].unique()) != len(cols):
            raise ValueError(f"Some columns are missing in the data 'T' for id = {i}")
        
        fakeSimPath = os.path.join(args.outputs_path, model_name, f"fake_simulated_sod_eod_{i}")

        with open(fakeSimPath, "wb") as f:
            pickle.dump((Ts, lob, lobL3), f)

    # save as 12D 
    paths = [i for i in os.listdir(os.path.join(os.path.join(args.outputs_path, model_name))) if ("fake_simulated" in i)]
    for p in paths:
        resPath = os.path.join(args.outputs_path, model_name, p)
        with open(resPath, 'rb') as f:
            results = pickle.load(f)
        
        ask_t = []
        bid_t = []
        ask_d = []
        bid_d= []
        event = []
        time = []
        for r, j in zip(results[1][1:], results[0][1:]):
            ask_t.append(r['Ask_touch'][0])
            bid_t.append(r['Bid_touch'][0])
            ask_d.append(r['Ask_deep'][0])
            bid_d.append(r['Bid_deep'][0])
            event.append(j[0])
            time.append(j[1])
        df = pd.DataFrame({"Time" : time, "event" : event, "Ask Price 1" : ask_t, "Bid Price 1": bid_t, "Ask Price 2": ask_d, "Bid Price 2" : bid_d})
        df['BidDiff'] = df['Bid Price 1'].diff()
        df['AskDiff'] = df['Ask Price 1'].diff()
        df['BidDiff2']= df['Bid Price 2'].diff()
        df['AskDiff2']= df['Ask Price 2'].diff()
        id = (resPath.split("/")[-1]).split("_")[-1]
        df["Date"] = id
        df.to_csv(os.path.join(args.outputs_path, model_name, f"fake_{id}_12D.csv"))