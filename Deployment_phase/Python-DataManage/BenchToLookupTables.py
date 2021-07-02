import argparse
import os
import fnmatch
import pandas as pd

parser = argparse.ArgumentParser(description='Translate Deployment Phase Microbenchmark Outputs to Lookup tables for the Autotuning Runtime')
parser.add_argument("-i", "--inputdir", type=str,help="Deployment Phase's microbenchmark output directory")
parser.add_argument("-f", "--function", type=str, help="The name of the function (e.g. BLAS routine)")
parser.add_argument("-d", "--dev_id", type=int, default=0, help="The GPU device's ID")
parser.add_argument("-l", "--libID", type=int, help="The ID of the backened library used.")

args = parser.parse_args()

InDataDir = args.inputdir
func = args.function
dev_id = args.dev_id
libID= args.libID

if (libID == 0): # CuCuBLAS
	infile_libname = 'cublas'
else:
	exit("Invalid libID")


entries = os.listdir(InDataDir)
outdata = pd.DataFrame()
for filename in entries:
	if fnmatch.fnmatch(filename, 'cublas%s_dev-%d*.log' %(func,dev_id)):
		print(filename)	
		parts = filename.split('.')[0].split('_')
		indata = pd.read_csv(InDataDir+'/'+filename, header = None)
		if len(parts) == 2:
			print('%s : no flags' %filename)
			outdata.append(indata, verify_integrity=True)
		else: 
			flags = parts[2:]
			print(flags)
			for flag in reversed(flags):
				flagdata = flag.split('-')
				print('Flagname: %s, flagvalue: %s' %(flagdata[0], flagdata[1]))
				indata.insert(0, flagdata[0], flagdata[1])
			#print(indata)
			if outdata.empty:
				outdata = indata.copy()
			else:
				outdata = pd.concat([outdata,indata], ignore_index=True) #outdata.append(indata, ignore_index=True, verify_integrity=True)
		#[-1].split('-')[1]
		#print(indata)
final_output = outdata.sort_values(0)
final_output.to_csv("%s/../Database/%s_lookup-table_dev-%d.log" %(InDataDir, func, dev_id), index = False, header = False, float_format = '%e')		

#outfile_h2d_name = sprintf("%s/../Database/Linear-Model_to-%d_from--1.log", InDataDir, dev_id)
#outfile_d2h_name = sprintf("%s/../Database/Linear-Model_to--1_from-%d.log", InDataDir, dev_id)

