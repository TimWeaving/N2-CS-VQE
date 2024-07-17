import numpy as np
from matplotlib import pyplot as plt

# c1 = plt.cm.plasma(np.log10((np.math.comb(4, 1)**2)/(np.math.comb(10,7)**2))/np.log10(1/(np.math.comb(10,7)**2)))
# c2 = plt.cm.plasma(np.log10((np.math.comb(5, 2)**2)/(np.math.comb(10,7)**2))/np.log10(1/(np.math.comb(10,7)**2)))
# c3 = plt.cm.plasma(np.log10((np.math.comb(6, 3)**2)/(np.math.comb(10,7)**2))/np.log10(1/(np.math.comb(10,7)**2)))
# c4 = plt.cm.plasma(np.log10((np.math.comb(7, 4)**2)/(np.math.comb(10,7)**2))/np.log10(1/(np.math.comb(10,7)**2)))

c0 = plt.cm.plasma(0)
c1 = plt.cm.plasma(1/5)
c2 = plt.cm.plasma(2/5)
c3 = plt.cm.plasma(3/5)
c4 = plt.cm.plasma(4/5)
c5 = plt.cm.plasma(0.999)

plt_styles = {
    'CASCI(4,2)': {'label':'CASCI    (4o,2e)','lw':1,'ls':'--','color':c4},
    'CASSCF(4,2)':{'label':'CASSCF (4o,2e)',  'lw':1,'ls':':', 'color':c4},
    'CASCI(5,4)': {'label':'CASCI    (5o,4e)','lw':1,'ls':'--','color':c3},
    'CASSCF(5,4)':{'label':'CASSCF (5o,4e)',  'lw':1,'ls':':', 'color':c3},
    'CASCI(6,6)': {'label':'CASCI    (6o,6e)','lw':1,'ls':'--','color':c2},
    'CASSCF(6,6)':{'label':'CASSCF (6o,6e)',  'lw':1,'ls':':', 'color':c2},
    'CASCI(7,8)': {'label':'CASCI    (7o,8e)','lw':1,'ls':'--','color':c1},
    'CASSCF(7,8)':{'label':'CASSCF (7o,8e)',  'lw':1,'ls':':', 'color':c1},
    'FCI':    {'label':'FCI',    'lw':1.0,'ls':'-','color':'black','zorder':100},
    'HF':     {'label':'HF',     'lw':0.7,'ls':'-','color':c5},
    'MP2':    {'label':'MP2',    'lw':0.7,'ls':'-','color':c4},
    'CISD':   {'label':'CISD',   'lw':0.7,'ls':'-','color':c3},
    'CCSD':   {'label':'CCSD',   'lw':0.7,'ls':'-','color':c2},
    'CCSD(T)':{'label':'CCSD(T)','lw':0.7,'ls':'-','color':c1},
    'NC':     {'label':'Noncontextual','lw':1,'ls':'-','color':'blue'},
    'CS':     {'label':'CS-VQE (5q)','lw':1,'ls':'','color':c0,
               'marker':'.','capsize':3,'ms':7,'zorder':200},
}