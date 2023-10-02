import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from pylab import cm
from collections import OrderedDict
import math
from scipy import stats

import plot_stuff as mplt 

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

font=8

# width as measured in inkscape
width = 7.2
height = width / 1.618 #*3/2

colors_fat= mplt.cp3[0]
colors=[mplt.cp3[1],mplt.cp3[2]]
colors_thin = 'orange'

def validation_set_split_BLAS3(name,locs,mid_sizes,ctrs_fat,ctrs_thin,input_set):
	cond_total = False
	for loc in locs:
		cond_loc = ((input_set['Aloc'] ==  loc[0] ) & (input_set['Bloc'] ==  loc[1]) & (input_set['Cloc'] ==  loc[2]))
		for mid_size in mid_sizes:
			cond_sz = ((input_set['K']*input_set['M']*input_set['N'] <= (mid_size**3)) & ((input_set['K']+1)*(input_set['M']+1)*(input_set['N']+1) >= (mid_size-1)**3))
			for ctr_fat in ctrs_fat:
				#set_row = input_set[(input_set['Aloc'] ==  loc[0] ) & (input_set['Bloc'] ==  loc[1]) & (input_set['Cloc'] ==  loc[2]) & 
							#(input_set['K']*input_set['M']*input_set['N'] <= (mid_size**3)) & ((input_set['K']+1)*(input_set['M']+1)*(input_set['N']+1) >= (mid_size-1)**3) &
				cond_fat = ((input_set['M'] == input_set['N']) & (input_set['N'] >= input_set['K']*(ctr_fat**3)/8) & (input_set['N'] <= (input_set['K']+1)*(ctr_fat**3)/8) )
				cond_total = cond_total | (cond_loc & cond_sz & cond_fat)
				#set_row = input_set[cond_loc & cond_sz & cond_fat]
				#print (set_row)
			for ctr_thin in ctrs_thin:
				cond_thin = ((input_set['M'] == input_set['N']) & (input_set['N'] >= input_set['K']*8/(ctr_thin**3)) & (input_set['N'] <= (input_set['K']+1)*8/(ctr_thin**3)) )
				cond_total = cond_total | (cond_loc & cond_sz & cond_thin)
				#set_row = input_set[cond_loc & cond_sz & cond_thin]
				#print (set_row)

	return input_set[cond_total]

machine_names= ['testbed-I_Tesla-K40', 'testbed-II_Tesla-V100',] #
cool_names= ['testbed-I', 'testbed-II'] 
machine_plot_lims = [(0.4, 1.4), (0,4), (1,8.5), (2.8,17)] # 
funcs=['Dgemm','Sgemm']
version='final'
mid_sizes= []
for i in range(4096,16384+1,1024):
	mid_sizes.append(i) #4096,8192,12288,16384]
print(mid_sizes)
#symbolist=['-s','--^','-.o']
ctr = 0

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=font-2)
plt.rc('ytick', labelsize=font-2)
plt.rc('axes', labelsize=font-2)
plt.grid(True)
fig, axs = plt.subplots( len(funcs), len(machine_names))

ctr = 0
for machine_num in range(len(machine_names)):
	functr = 0
	machine = machine_names[machine_num]
	for func in funcs:

		eval_CoCo = pd.read_csv('../Results/%s/evaluation/CoCopeLia_%s_0_v%s.log' % (machine, func, version),
							   header = None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T_CoCo', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
		eval_BLASX = pd.read_csv('../Results/%s/evaluation/BLASX_%s_0.log' % (machine, func),
							   header = None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T_BLASX', 'Aloc','Bloc', 'Cloc',  'BLASX_t'])
		eval_cuBLASXt = pd.read_csv('../Results/%s/evaluation/cuBLASXt_%s_0.log' % (machine, func),
							   header = None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T_cuBLAS', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

		print( "Read %d values from \"../Results/%s/evaluation/CoCopeLia_%s_0_v%s.log\"..." %( len(eval_CoCo), machine, func, version))
		print( "Read %d values from \"../Results/%s/evaluation/BLASX_%s_0.log\".." %( len(eval_BLASX), machine, func))
		print( "Read %d values from \"../Results/%s/evaluation/cuBLASXt_%s_0.log\".." %( len(eval_cuBLASXt), machine, func))

		validata_0 = pd.merge(eval_CoCo, eval_BLASX, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])
		validata = pd.merge(validata_0, eval_cuBLASXt, on = ['M', 'N', 'K', 'Aloc', 'Bloc', 'Cloc'])
		print( "Combined in dataframe of length = %d with head:" %( len(validata)))
		#print(validata.head(1))
		#print(validata['CoCopeLia_t'].max())
		#
		locs= [[0,0,1],[1,1,1]]
		ctrs_fat = [2]#[2,4,5,6,7,8]
		ctrs_thin = [] #[5,7,10,13,16]
		validata_sq = validation_set_split_BLAS3("Sq_set",locs,mid_sizes,ctrs_fat,ctrs_thin,validata)
		print( "Kept %d square values" %( len(validata_sq)))
		for loc_num in range(len(locs)):
			loc = locs[loc_num]
			validata_plot = validata_sq[((validata_sq['Aloc'] == loc[0]) & (validata_sq['Bloc'] == loc[1]) & (validata_sq['Cloc'] == loc[2]))]	
			print( "Plotting line for loc=[%d,%d,%d] with %d points..." %( loc[0], loc[1], loc[2], len(validata_plot)))
			streamed_list = list(validata_plot['CoCopeLia_t'])
			cublasXt_list = list(validata_plot['cublasXt_t'])
			BLASX_list = list(validata_plot['BLASX_t'])
			flop_list = dgemm_flops(validata_plot['M'], validata_plot['N'],validata_plot['K'])
			xaxis_list = list( map( lambda x: int(x**(1./3.)/1000), validata_plot['M']*validata_plot['N']*validata_plot['K']))
			_, streamed_list = zip(*sorted(zip(xaxis_list, streamed_list), key=lambda t: t[0]))
			_, cublasXt_list = zip(*sorted(zip(xaxis_list, cublasXt_list), key=lambda t: t[0]))
			_, BLASX_list = zip(*sorted(zip(xaxis_list, BLASX_list), key=lambda t: t[0]))
			xaxis_list, flop_list = zip(*sorted(zip(xaxis_list, flop_list), key=lambda t: t[0]))	
			if (loc[0] == loc[1] == loc[2] == 1):
				labeloc = 'Data:CPU mem'
			else:
				labeloc = 'C:CPU mem'
			axs[functr,ctr].plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, cublasXt_list)), 
			'--^', alpha=1, linewidth = 0.5,  markersize = 3, color=colors[loc_num], label='cuBLASXt Sq. %s' % (labeloc) )

			axs[functr,ctr].plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, BLASX_list)), 
			'--*', alpha=1, linewidth = 0.5, markersize = 3, color=colors[loc_num], label='BLASX Sq. %s' % (labeloc))

			axs[functr,ctr].plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, streamed_list)), 
			'-', alpha=1, linewidth = 1, color=colors[loc_num], label='CoCoPeLia Sq. %s' % (labeloc))
			#axs[functr,ctr].get_legend().remove()
			axs[functr,ctr].set_xlabel('')

		ctrs_fat = [4]	
		validata_fat = validation_set_split_BLAS3("Fat_set",[[1,1,1]],mid_sizes,ctrs_fat,[],validata)
		print( "Kept %d fat values" %( len(validata_fat)))
		#validata_filtered = pd.concat([validata_sq, validata_fathin])
		#print( "Combined top %d sqfathin values" %( len(validata_filtered)))
		for fat_num in []:#range(len(ctrs_fat)):
			fat = ctrs_fat[fat_num] 
			validata_plot = validata_fat # This is a hack, onyl works for len = 0, but its late. 
			print( "Plotting line for fat=%d with %d points..." %( fat, len(validata_plot)))
			if len(validata_plot)==0:
				break
			streamed_list = list(validata_plot['CoCopeLia_t'])
			cublasXt_list = list(validata_plot['cublasXt_t'])
			BLASX_list = list(validata_plot['BLASX_t'])
			flop_list = dgemm_flops(validata_plot['M'], validata_plot['N'],validata_plot['K'])
			xaxis_list = list( map( lambda x: int(x**(1./3.)/1000), validata_plot['M']*validata_plot['N']*validata_plot['K']))

			_, streamed_list = zip(*sorted(zip(xaxis_list, streamed_list), key=lambda t: t[0]))
			_, cublasXt_list = zip(*sorted(zip(xaxis_list, cublasXt_list), key=lambda t: t[0]))
			_, BLASX_list = zip(*sorted(zip(xaxis_list, BLASX_list), key=lambda t: t[0]))
			xaxis_list, flop_list = zip(*sorted(zip(xaxis_list, flop_list), key=lambda t: t[0]))	
			
			axs[functr,ctr].plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, cublasXt_list)), 
			'--^', alpha=1, linewidth = 0.5,  markersize = 3, color=colors_thin, label='cuBLASXt Thin-by-Fat-%d' % (fat) )

			axs[functr,ctr].plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, BLASX_list)), 
			'--*', alpha=1, linewidth = 0.5, markersize = 3, color=colors_thin, label='BLASX Thin-by-Fat-%d' % (fat))

			axs[functr,ctr].plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, streamed_list)), 
			'-', alpha=1, linewidth = 1, color=colors_thin, label='CoCoPeLia Thin-by-Fat-%d' % (fat))
			#axs[functr,ctr].get_legend().remove()
			axs[functr,ctr].set_xlabel('')

		ctrs_thin = [4]
		validata_thin = validation_set_split_BLAS3("Thin_set",[[1,1,1]],mid_sizes,[],ctrs_thin,validata)
		print( "Kept %d thin values" %( len(validata_thin)))
		#print(validata_thin)
		#validata_filtered = pd.concat([validata_sq, validata_fathin])
		#print( "Combined top %d sqfathin values" %( len(validata_filtered)))
		for thin_num in range(len(ctrs_thin)):
			thin = ctrs_thin[thin_num] 
			validata_plot = validata_thin # This is a hack, onyl works for len = 0, but its late. 
			print( "Plotting line for thin=%d with %d points..." %( thin, len(validata_plot)))
			if len(validata_plot)==0:
				break
			streamed_list = list(validata_plot['CoCopeLia_t'])
			cublasXt_list = list(validata_plot['cublasXt_t'])
			BLASX_list = list(validata_plot['BLASX_t'])
			flop_list = dgemm_flops(validata_plot['M'], validata_plot['N'],validata_plot['K'])
			xaxis_list = list( map( lambda x: int(x**(1./3.)/1000), validata_plot['M']*validata_plot['N']*validata_plot['K']))
			#print(cublasXt_list)
			#print(xaxis_list)
			_, streamed_list = zip(*sorted(zip(xaxis_list, streamed_list), key=lambda t: t[0]))
			_, cublasXt_list = zip(*sorted(zip(xaxis_list, cublasXt_list), key=lambda t: t[0]))
			_, BLASX_list = zip(*sorted(zip(xaxis_list, BLASX_list), key=lambda t: t[0]))
			xaxis_list, flop_list = zip(*sorted(zip(xaxis_list, flop_list), key=lambda t: t[0]))	
			#print(cublasXt_list)
			#print(xaxis_list)

			labeloc = 'Data:CPU mem'

			axs[functr,ctr].plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, cublasXt_list)), 
			'--^', alpha=1, linewidth = 0.5,  markersize = 3, color=colors_fat, label='cuBLASXt Fat-by-thin-%d %s' % (thin*2, labeloc) )

			axs[functr,ctr].plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, BLASX_list)), 
			'--*', alpha=1, linewidth = 0.5, markersize = 3, color=colors_fat, label='BLASX Fat-by-thin-%d %s' % (thin*2, labeloc))

			axs[functr,ctr].plot(xaxis_list, list(map(lambda x,y: GigaVal_per_s(x,y*1024),flop_list, streamed_list)), 
			'-', alpha=1, linewidth = 1, color=colors_fat, label='CoCoPeLia Fat-by-thin-%d %s' % (thin*2, labeloc))
			#axs[functr,ctr].get_legend().remove()
			axs[functr,ctr].set_xlabel('')

		#ax.set_xlabel('Problem size (N*N)N')

		#ax.set_ylim(machine_plot_lims[ctr*len(machine_names) +machine_num])
		#axs[functr,ctr].grid(ls='--', linewidth = 0.6, color = 'gray')

		axs[functr,ctr].set_title(func+ '@' + cool_names[ctr], fontsize=font)

		#axs[functr,ctr].set_ylabel(func)
		if ((functr == 1) & (ctr == 0)):
			axs[functr,ctr].text( 15, 0, "Problem Size (N*N)*N", fontsize=font)
		#elif (ctr == 0):
			#axs[functr,ctr].set_ylabel(func)
		axs[functr,ctr].set_ylabel("Performance (Tflops/s)", fontsize=font)
		if (functr == 0):
			axs[functr,ctr].set_xticks([])

		#axs[functr,ctr].set_yscale('log')
		#axs[functr,ctr].grid()

		functr += 1


		validata['size'] = validata['M']*validata['N']*validata['K']
		full_offload_samples =  validata[( (validata['Aloc'] == 1) & (validata['Bloc'] == 1) & (validata['Cloc'] == 1)) ]
		partial_offload_samples = validata[((validata['Aloc'] == 1) | (validata['Bloc'] == 1) | (validata['Cloc'] == 1)) &
							((validata['Aloc'] == 0) | (validata['Bloc'] == 0) | (validata['Cloc'] == 0))]

		print('Function: %s Machine : %s' % (func, machine))

		# SOTA = State of the art
		full_offload_samples['SOTA'] = full_offload_samples[['cublasXt_t', 'BLASX_t']].min(axis=1)
		partial_offload_samples['SOTA'] = partial_offload_samples[['cublasXt_t', 'BLASX_t']].min(axis=1)

		#full_offload_CoCoBLAS_speedup = (full_offload_samples['SOTA'] - full_offload_samples['CoCopeLia_t'])/ full_offload_samples['SOTA']
		#print("CoCopeLia Full offload AR.MEAN[(t_baseline - t_mine)/t_baseline] : %lf" % (100*full_offload_CoCoBLAS_speedup.mean()))
		#partial_offload_CoCoBLAS_speedup = (partial_offload_samples['SOTA'] - partial_offload_samples['CoCopeLia_t']) / partial_offload_samples['SOTA']
		#print("CoCopeLia Partial offload AR.MEAN[(t_baseline - t_mine)/t_baseline]: %lf\n" % (100*partial_offload_CoCoBLAS_speedup.mean()))

		#print("Torsten Rule 4: Avoid summarizing ratios; summarize the costs or rates that the ratios base on instead.")
		#print("Only if these are not available use the geometric mean for summarizing ratios.")
		#print("CoCopeLia Full offload [AR.MEAN(t_baseline) - AR.MEAN(t_mine)]/AR.MEAN(t_baseline): %lf" % (100* 
		#	((full_offload_samples['SOTA'].mean() - full_offload_samples['CoCopeLia_t'].mean())/full_offload_samples['SOTA'].mean())))
		#print("CoCopeLia Partial offload [AR.MEAN(t_baseline) - AR.MEAN(t_mine)]/AR.MEAN(t_baseline): %lf\n\n" % (100* 
		#	((partial_offload_samples['SOTA'].mean() - partial_offload_samples['CoCopeLia_t'].mean())/partial_offload_samples['SOTA'].mean())))

		print("CoCopeLia Full offload 1 - GEO.MEAN[t_mine/t_baseline]: %lf" % (100* 
			(1 - stats.gmean(full_offload_samples['CoCopeLia_t']/full_offload_samples['SOTA'],axis=0))))
		print("CoCopeLia Partial offload 1 - GEO.MEAN[t_mine/t_baseline]: %lf\n\n" % (100* 
			(1 - stats.gmean(partial_offload_samples['CoCopeLia_t']/partial_offload_samples['SOTA'],axis=0))))
		#slowdown_list = []
		#speedup_list = []
		#for elem in full_offload_CoCoBLAS_speedup:
		#	if elem < 0:
		#		slowdown_list.append(elem)
		#	else:
		#		speedup_list.append(elem)

		#print("Full offload AR.MEAN Speedup CoCopeLia: %lf " % (100*stats.gmean(full_offload_CoCoBLAS_speedup,axis=0))) # arithmetic : full_offload_CoCoBLAS_speedup.mean()
		#print("Partial offload AR.MEAN Speedup CoCopeLia: %lf " % (100*stats.gmean(partial_offload_CoCoBLAS_speedup,axis=0))) # arithmetic : partial_offload_CoCoBLAS_speedup.mean()


	ctr += 1

#Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#	ax.label_outer()
#fig.subplots_adjust(left=.10, bottom=.06, right=.99, top=.88)
fig.subplots_adjust(left=.06, bottom=.08, right=.99, top=.84)
fig.set_size_inches(width, height)
# Create the legend
handles, labels = fig.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
	loc="upper center",   # Position of legend
	borderaxespad=0.1,    # Small spacing around legend box
	#title="Model",  # Title for the legend
	fontsize=font, fancybox = False, ncol=3
	)


fig.savefig('./Plots/Evaluation_CoCopeLia_plot_all.pdf' )

