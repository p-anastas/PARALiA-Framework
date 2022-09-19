
import argparse
import os
import fnmatch
import pandas as pd

parser = argparse.ArgumentParser(description='Translate Deployment Phase Microbenchmark Outputs to Lookup tables for the Autotuning Runtime')
parser.add_argument("-i", "--inputdir", type=str,help="Deployment Phase's microbenchmark output directory")
parser.add_argument("-o", "--outputdir", type=str,help="Lookup table store directory")
parser.add_argument("-f", "--function", type=str, help="The name of the function (e.g. BLAS routine)")
parser.add_argument("-d", "--dev_id", type=int, default=0, help="The GPU device's ID")
parser.add_argument("-l", "--libID", type=int, help="The ID of the backened library used.")

args = parser.parse_args()

InDataDir = args.inputdir
OutDataDir = args.outputdir
func = args.function
dev_id = args.dev_id
libID= args.libID

if (libID == 0): # CuCuBLAS
	infile_libname = 'cublas'
else:
	exit("Invalid libID")

version='1.3'


entries = os.listdir(InDataDir)
outdata = pd.DataFrame()
for filename in entries:
	if fnmatch.fnmatch(filename, 'cublas%s_dev-%d*_%s.log' %(func,dev_id, version)):
		print(filename)	
		parts = filename.split('_')[0:-1]
		indata = pd.read_csv(InDataDir+'/'+filename, header = None)
		if len(parts) == 2:
			print('%s : no flags' %filename)
			outdata = pd.concat([outdata,indata], ignore_index=True)
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
				outdata = pd.concat([outdata,indata], ignore_index=True)
		#[-1].split('-')[1]
		#print(indata)
print(outdata)
final_output = outdata.sort_values(0)
final_output.to_csv("%s/%s_lookup-table_dev-%d.log" %(OutDataDir, func, dev_id), index = False, header = False, float_format = '%e')		
