#include "SteerForEvasionCUDA.cuh"

extern "C"
{
	__global__ void SteerForEvasionKernel(	// Agent data.
											float4 const*	pdPosition,
											float4 const*	pdDirection,
											float4 *		pdSteering,

											float3 const	menacePosition,
											float3 const	menaceDirection,
											float const		menaceSpeed,
											
											float const		maxPredictionTime,

											size_t const	numAgents,

											float const		fWeight,
											uint *			pdAppliedKernels,
											uint const		doNotApplyWith
										  );
}

using namespace OpenSteer;

SteerForEvadeCUDA::SteerForEvadeCUDA( AgentGroup * pAgentGroup, float3 const& menacePosition, float3 const& menaceDirection, float const menaceSpeed, float const fMaxPredictionTime, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_fMaxPredictionTime( fMaxPredictionTime ),
	m_menacePosition( menacePosition ),
	m_menaceDirection( menaceDirection ),
	m_menaceSpeed( menaceSpeed )
{
	// Nothing to do.
}

void SteerForEvadeCUDA::init( void )
{
	// Nothing to do.
}

void SteerForEvadeCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required device data.
	float4 const*	pdPosition			= m_pAgentGroupData->pdPosition();
	float4 const*	pdDirection			= m_pAgentGroupData->pdDirection();
	float4 *		pdSteering			= m_pAgentGroupData->pdSteering();

	uint const		numAgents			= getNumAgents();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	SteerForEvasionKernel<<< grid, block >>>(	// Agent data.
												pdPosition,
												pdDirection,
												pdSteering,

												m_menacePosition,
												m_menaceDirection,
												m_menaceSpeed,

												m_fMaxPredictionTime,

												numAgents,
												m_fWeight,
												pdAppliedKernels,
												m_doNotApplyWith
												);
	cutilCheckMsg( "SteerForEvasionKernel failed." );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForEvadeCUDA::close( void )
{
	// Agent group data may have changed.
	m_pAgentGroup->SetSyncHost();
}
