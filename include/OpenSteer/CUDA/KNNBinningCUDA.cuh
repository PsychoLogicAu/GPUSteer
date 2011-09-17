#ifndef OPENSTEER_KNNBINNING_CUH
#define OPENSTEER_KNNBINNING_CUH

#include "AbstractCUDAKernel.cuh"

#include "KNNBinData.cuh"
#include "KNNData.cuh"

namespace OpenSteer
{
class KNNBinningUpdateDBCUDA : public AbstractCUDAKernel
{
protected:
	KNNBinData *	m_pKNNBinData;

public:
	KNNBinningUpdateDBCUDA( AgentGroup * pAgentGroup, KNNBinData * pKNNBinData );
	virtual ~KNNBinningUpdateDBCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};

class KNNBinningCUDA : public AbstractCUDAKernel
{
protected:
	KNNData *		m_pKNNData;
	KNNBinData *	m_pKNNBinData;
	BaseGroup *		m_pOtherGroup;

public:
	KNNBinningCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinData * pKNNBinData, BaseGroup * pOtherGroup );
	virtual ~KNNBinningCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );




};	// class KNNBinningCUDA
}	// namespace OpenSteer


#endif