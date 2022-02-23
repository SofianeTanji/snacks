from benchmark_functions import *

print("Cerr VS M")
arrm = np.linspace(100,3000,40)
cerr_vs_m("a9a", 0.8, 1e-3, 35000, 5, 1e-6, 1., arrm)

print("Cerr VS M")
arrl = np.logspace(-9, -1, 40)
cerr_vs_lmbd("a9a", 0.8, 1e-3, 35000, 5, 1000, 1., arrl)

print("Combined behaviour")
arr_m = np.array([10, 20, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000])
arr_l = np.array(
    [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
)
cerr_m_lmbd("a9a", 0.8, 1e-3, 35000, 5, 1., arr_m, arr_l)

print("Combined behaviour KRR")
heatmap("a9a", 0.8, 1e-3, 5, arr_m, arr_l)

print("Grid Search Snacks")
arrg = np.logspace(-5, -1, 12)
grid_search_map("a9a", 0.8, 35000, 1000, 5, arrg, arr_l)

print("Grid Search KRR")
grid_search_krr("a9a", 0.8, 1000, 5, arrg, arr_l)

print("Objective value vs It")
obj_vs_it("a9a", 0.8, 1e-3, 1000, 1e-6, 1., 25000)