///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The "Asset" related function implementations.
///

#include "Operative.hpp"
#include "unihelpers.hpp"

Operative* CoCoAsignTilesToOperativesGemm(Asset2D<double>* A_asset, Asset2D<double>* B_asset, Asset2D<double>* C_asset, int T, int* kernelNum){

	short lvl = 4;
	/// Generalize for not exact tiles
	size_t Mlast = kernel->Ms%MblockSz, Nlast = kernel->Ns%NblockSz, Klast= kernel->Ks%KblockSz;
	kernel->MblockSz = MblockSz;
	kernel->NblockSz = NblockSz;
	kernel->KblockSz = KblockSz;
	kernel->MgridSz = kernel->Ms/MblockSz;
	kernel->NgridSz = kernel->Ns/NblockSz;
	kernel->KgridSz = kernel->Ks/KblockSz;
	if (Mlast > MblockSz/4) kernel->MgridSz++;
	else Mlast+=MblockSz;
	if (Nlast > NblockSz/4) kernel->NgridSz++;
	else Nlast+=NblockSz;
	if (Klast > KblockSz/4) kernel->KgridSz++;
	else Klast+=KblockSz;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCopeLiaDgemmSubkernelCreateGrid(kernel,%zu,%zu,%zu)\n", MblockSz, NblockSz, KblockSz);
	lprintf(lvl,"MgridSz = %zu, NgridSz = %zu, KgridSz = %zu\n", kernel->MgridSz, kernel->NgridSz, kernel->KgridSz);
	lprintf(lvl,"Mlast = %zu, Nlast = %zu, Klast = %zu\n", Mlast, Nlast, Klast);
#endif

	size_t current_ctr, ptr_offset, gpu_ptr_offset, out_ptr_offset;

	*kernelNum = kernel->MgridSz*kernel->NgridSz*kernel->KgridSz;

	kernel3_p* kernels = (kernel3_p*) malloc(*kernelNum*sizeof(kernel3_p));

	size_t MtempSz = kernel->MblockSz, NtempSz = kernel->NblockSz, KtempSz = kernel->KblockSz;

	for (int mi = 0; mi < kernel->MgridSz; mi++)
	{
		if ( mi == kernel->MgridSz - 1) MtempSz = Mlast;
		else MtempSz = kernel->MblockSz;
		for (int ni = 0 ; ni < kernel->NgridSz; ni++){
			if ( ni == kernel->NgridSz - 1) NtempSz = Nlast;
			else NtempSz = kernel->NblockSz;
			for (int ki = 0; ki < kernel->KgridSz; ki++){
        			if ( ki == kernel->KgridSz - 1) KtempSz = Klast;
				else KtempSz = kernel->KblockSz;
        			current_ctr = mi*kernel->NgridSz*kernel->KgridSz + ni*kernel->KgridSz + ki;
				kernels[current_ctr] = CoCopeLiaDgemmSubkernelClone(kernel, mi, ni, ki, MtempSz, NtempSz, KtempSz);
			}
		}
	}
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
	return kernels;
}
