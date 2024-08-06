import sys
sys.path.append("/home/konajain/code/lobSimulations")
from hawkes.simulate_smalltick import run as run

def main(i):
    # ergodic limit
    if i <= 10:
        run('erg_spr0_'+str(5+10*i), spread0 = 5+10*i)
    # M_med:
    if (i > 10) and (i <= 20):
        M_med = 5*(i - 10)
        run('erg_Mmed_'+str(M_med), M_med = M_med)
    # Avg spread, spreadBeta :
    if (i > 20) and (i <= 26):
        avgSpr = [0.010, 0.016, 0.05, 0.10, 0.50, 1.00][i - 21]
        beta = [0.95, 0.75, 0.7 , 0.6, 0.55, 0.5][i - 21]
        run('tickness_avgSpr_beta_'+str(avgSpr)+'_'+str(beta), avgSpread= avgSpr, beta=beta)
    # Pis:
    if (i > 26) and (i <= 32):
        Pis = {'lo_deep_Bid': [[0.01, 0.05, 0.1, 0.2, 0.5, 0.8][i-26],
                               []],
               'lo_inspread_Bid': [[0.01, 0.05, 0.1, 0.2, 0.5, 0.8][i-26],
                                   []],
               'lo_top_Bid': [[0.01, 0.05, 0.1, 0.2, 0.5, 0.8][i-26],
                              []],
               'mo_Bid': [0.1,
                          []]}
        Pis["mo_Ask"] = Pis["mo_Bid"]
        Pis["lo_top_Ask"] = Pis["lo_top_Bid"]
        Pis["co_top_Ask"] = Pis["lo_top_Ask"]
        Pis["co_top_Bid"] = Pis["lo_top_Bid"]
        Pis["lo_inspread_Ask"] = Pis["lo_inspread_Bid"]
        Pis["lo_deep_Ask"] = Pis["lo_deep_Bid"]
        Pis["co_deep_Ask"] = Pis["lo_deep_Ask"]
        Pis["co_deep_Bid"] = Pis["lo_deep_Bid"]
        run('erg_mo_lo_ratio_'+str(0.1/([0.01, 0.05, 0.1, 0.2, 0.5, 0.8][i-26])), Pis=Pis)
    # M0:
    if (i > 32) and (i <= 39):
        Pi_M0={'m_T': [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 0.99][i - 33], 'm_D' : [ 0.0005, 0.001, 0.01, 0.1, 0.2, 0.5, 0.9][i - 33]}
        run('erg_M0_'+str(Pi_M0['m_T']), Pi_M0 = Pi_M0)
    # eta
    if (i > 39) and (i<=49):
        Pi_eta = {'eta_T' : [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 0.99][i-40],
                  'eta_IS' : [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 0.99][i-40],
                  'eta_T+1': [ 0.0005, 0.001, 0.01, 0.1, 0.2, 0.5, 0.9][i-40]}
        run('erg_eta_'+str(Pi_eta['eta_T']), Pi_eta = Pi_eta)
    return 0


main( int(sys.argv[1]) - 1 )