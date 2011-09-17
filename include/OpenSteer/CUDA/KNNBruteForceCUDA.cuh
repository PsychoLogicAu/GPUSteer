#ifndef OPENSTEER_KNNBRUTEFORCECUDA_CUH
#define OPENSTEER_KNNBRUTEFORCECUDA_CUH

#include "AbstractCUDAKernel.cuh"

#include "KNNData.cuh"
#include "KNNDatabase.cuh"

namespace OpenSteer
{
class KNNBruteForceCUDA : public AbstractCUDAKernel
{
protected:
	// Temporary device storage.
	float *		m_pdDistanceMatrix;
	uint *		m_pdIndexMatrix;

	KNNData *	m_pKNNData;
	BaseGroup *	m_pOtherGroup;

public:
	KNNBruteForceCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, BaseGroup * pOtherGroup );
	virtual ~KNNBruteForceCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );

};	// class KNNBruteForceCUDA

class KNNBruteForceCUDAV2 : public AbstractCUDAKernel
{
protected:
	KNNData *	m_pKNNData;
	BaseGroup *	m_pOtherGroup;

public:
	KNNBruteForceCUDAV2( AgentGroup * pAgentGroup, KNNData * pKNNData, BaseGroup * pOtherGroup );
	virtual ~KNNBruteForceCUDAV2( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class KNNBruteForceCUDAV2

class KNNBruteForceCUDAV3 : public AbstractCUDAKernel
{
protected:
	KNNData *	m_pKNNData;
	BaseGroup *	m_pOtherGroup;

public:
	KNNBruteForceCUDAV3( AgentGroup * pAgentGroup, KNNData * pKNNData, BaseGroup * pOtherGroup );
	virtual ~KNNBruteForceCUDAV3( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class KNNBruteForceCUDAV3
}	// namespace OpenSteer

#endif
