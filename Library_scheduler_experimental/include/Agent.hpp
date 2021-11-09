/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The header containing the "Agent" (== per-device scheduler) definition for data scheduling and management in heterogeneous multi-device systems. 
///

#ifndef AGENT_H
#define AGENT_H

#include<iostream>
#include <string>

#include <unihelpers.hpp>
#include "Asset.hpp"

class Operative;

class Agent
{
	private:
	public:
		std::string name;
		Operative* operatives;
		void* local_data_bank;
		void* Asset1D_list;
		void* Asset2D_list;
		void check_local_data_bank();
		void* operation;
		void print() { std::cout << "Agent : " << name; }

};

#endif
