#include "KNNBinningV1.cuh"

using namespace OpenSteer;

#include "KNNBinData.cuh"

#include <thrust/sort.h>

//#include "DebugUtils.h"

#define TIMING

// Kernel file function prototypes.
extern "C"
{
	// Bind texCellIndices to the cudaArray.
	__host__ void KNNBinningV1BindTexture( cudaArray * pCudaArray );
	__host__ void KNNBinningV1UnbindTexture( void );

	__host__ void KNNBinningV1KernelBindTextures(			uint const*		pdBCellStart,
														uint const*		pdBCellEnd,
														uint const*		pdBIndices,
														float4 const*	pdBPositionSorted,
														uint const		numCells,
														uint const		numB
														);

	__host__ void KNNBinningV1KernelUnbindTextures( void );
	__host__ void KNNBinningV1ReorderDBBindTextures(	float4 const*	pdPosition,
													uint const		numAgents );
	__host__ void KNNBinningV1ReorderDBUnbindTextures( void );

	__global__ void KNNBinningV1BuildDB(					float4 const*	pdPosition,				// In:	Positions of each agent.
														size_t *		pdAgentIndices,			// Out:	Indices of each agent.
														size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
														size_t const	numAgents
														);

	__global__ void KNNBinningV1ReorderDB(				uint const*		pdAgentIndices,		// In: (sorted) agent index.
														uint const*		pdCellIndices,		// In: (sorted) cell index agent is in.

														float4 *		pdPositionSorted,	// Out: Sorted agent positions.

														uint *			pdCellStart,		// Out: Start index of this cell in pdCellIndices.
														uint *			pdCellEnd,			// Out: End index of this cell in pdCellIndices.

														size_t const	numAgents
														);

	__global__ void KNNBinningV1Kernel(					// Group A
														float4 const*	pdAPositionSorted,			// In:	Sorted group A positions.
														uint const*		pdAIndices,					// In:	Sorted group A indices
														uint const		numA,						// In:	Size of group A.

														// Cell neighbor info.
														int const		radius,						// In:	Search radius (in cells) to consider.

														// Output data.
														uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
														float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.
														uint const		k,							// In:	Number of neighbors to consider.

														uint const		numB,						// In:	Size of group B.
														bool const		groupWithSelf				// In:	Are we testing this group with itself? (group A == group B)
														);
}

#pragma region KNNBinningV1UpdateDBCUDA

KNNBinningV1UpdateDBCUDA::KNNBinningV1UpdateDBCUDA( BaseGroup * pGroup, KNNBinData * pKNNBinData )
:	AbstractCUDAKernel( NULL, 1.f, 0 ),
	m_pGroup( pGroup ),
	m_pKNNBinData( pKNNBinData )
{
	// Nothing to do.
}

void KNNBinningV1UpdateDBCUDA::init( void )
{
	// Bind the lookup texture.
	KNNBinningV1BindTexture( m_pKNNBinData->pdCellIndexArray() );
}

void KNNBinningV1UpdateDBCUDA::run( void )
{
	dim3 grid	= dim3( (m_pGroup->Size() + THREADSPERBLOCK - 1) / THREADSPERBLOCK );
	dim3 block	= dim3( THREADSPERBLOCK );

	// Gather required data.
	float4 const*	pdPosition				= m_pGroup->pdPosition();
	
	uint *			pdCellIndices			= m_pGroup->GetKNNDatabase().pdCellIndices();

	uint *			pdAgentIndicesSorted	= m_pGroup->GetKNNDatabase().pdAgentIndicesSorted();
	uint *			pdCellIndicesSorted		= m_pGroup->GetKNNDatabase().pdCellIndicesSorted();

	float4 *		pdPositionSorted		= m_pGroup->GetKNNDatabase().pdPositionSorted();
	uint *			pdCellStart				= m_pGroup->GetKNNDatabase().pdCellStart();
	uint *			pdCellEnd				= m_pGroup->GetKNNDatabase().pdCellEnd();

	uint const&		numAgents				= m_pGroup->Size();

#if defined TIMING
	//
	//	TIMING: hard to get exact times with profiling, too many operations.
	//
	// Events for timing the complete operation.
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );
#endif

	// Call KNNBinningBuildDB to build the database. 
	KNNBinningV1BuildDB<<< grid, block >>>( pdPosition, pdAgentIndicesSorted, pdCellIndices, numAgents );
	cutilCheckMsg( "KNNBinningBuildDB failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Copy pdCellIndices to  pdCellIndicesSorted.
	CUDA_SAFE_CALL( cudaMemcpy( pdCellIndicesSorted, pdCellIndices, numAgents * sizeof(uint), cudaMemcpyDeviceToDevice ) );

	// Sort pdAgentIndicesSorted on pdCellIndicesSorted using thrust.
	thrust::sort_by_key(	thrust::device_ptr<uint>( pdCellIndicesSorted ),
							thrust::device_ptr<uint>( pdCellIndicesSorted + numAgents ),
							thrust::device_ptr<uint>( pdAgentIndicesSorted ) );

	// Set all cells to empty.
	CUDA_SAFE_CALL( cudaMemset( pdCellStart, 0xffffffff, m_pKNNBinData->getNumCells() * sizeof(uint) ) );

	// Bind the textures.
	KNNBinningV1ReorderDBBindTextures( pdPosition, numAgents );

	// Call KNNBinningReorderDB to re-order the data in the DB.
	KNNBinningV1ReorderDB<<< grid, block >>>( pdAgentIndicesSorted, pdCellIndicesSorted, pdPositionSorted, pdCellStart, pdCellEnd, numAgents );	cutilCheckMsg( "KNNBinningReorderDB failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	KNNBinningV1ReorderDBUnbindTextures();

#if defined TIMING
	//
	//	TIMING:
	//
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	char szString[128] = {0};
	sprintf_s( szString, "KNNBinningV1UpdateDBCUDA,%f\n", elapsedTime );
	//OutputDebugStringToFile( szString );
	OutputDebugString( szString );

	// Destroy the events.
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
#endif
}

void KNNBinningV1UpdateDBCUDA::close( void )
{
	// Unbind the texture.
	KNNBinningV1UnbindTexture();

	// The AgentGroup's database has now changed.
	m_pGroup->SetSyncHost();
}
#pragma endregion


#pragma region KNNBinningV1CUDA

KNNBinningV1CUDA::KNNBinningV1CUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinData * pKNNBinData, BaseGroup * pOtherGroup, uint const searchRadius )
:	AbstractCUDAKernel( pAgentGroup, 1.f, 0 ),
	m_pKNNData( pKNNData ),
	m_pKNNBinData( pKNNBinData ),
	m_pOtherGroup( pOtherGroup ),
	m_searchRadius( searchRadius )
{
}

void KNNBinningV1CUDA::init( void )
{
	// Bind the cell indices texture.
	//KNNBinningV1CUDABindTexture( m_pKNNBinData->pdCellIndexArray() );
}

void KNNBinningV1CUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather the required data.
	float4 const*		pdAPositionSorted		= m_pAgentGroup->GetKNNDatabase().pdPositionSorted();
	uint const*			pdAIndices				= m_pAgentGroup->GetKNNDatabase().pdAgentIndicesSorted();
	uint const&			numA					= getNumAgents();

	float4 const*		pdBPositionSorted		= m_pOtherGroup->GetKNNDatabase().pdPositionSorted();
	uint const*			pdBIndices				= m_pOtherGroup->GetKNNDatabase().pdAgentIndicesSorted();
	uint const&			numB					= m_pOtherGroup->Size();

	uint const*			pdBCellStart			= m_pOtherGroup->GetKNNDatabase().pdCellStart();
	uint const*			pdBCellEnd				= m_pOtherGroup->GetKNNDatabase().pdCellEnd();
	uint const&			numCells				= m_pKNNBinData->getNumCells();

	uint *				pdKNNIndices			= m_pKNNData->pdKNNIndices();
	float *				pdKNNDistances			= m_pKNNData->pdKNNDistances();
	uint const&			k						= m_pOtherGroup->GetKNNDatabase().k();

	bool const			groupWithSelf			= m_pAgentGroup == m_pOtherGroup;

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(float) + sizeof(uint));

#if defined TIMING
	//
	//	TIMING: hard to get exact times with profiling, too many operations.
	//
	// Events for timing the complete operation.
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );
#endif

	// Bind the textures.
	KNNBinningV1KernelBindTextures(	pdBCellStart, pdBCellEnd, pdBIndices, pdBPositionSorted, numCells, numB );

	KNNBinningV1BindTexture( m_pKNNBinData->pdCellIndexArray() );

	// Call the KNNBinning kernel.
	KNNBinningV1Kernel<<< grid, block, shMemSize >>>(	pdAPositionSorted,
													pdAIndices,
													numA,
													m_searchRadius,
													pdKNNIndices,
													pdKNNDistances,
													k,
													numB,
													groupWithSelf
													);
	cutilCheckMsg( "KNNBinningKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	KNNBinningV1KernelUnbindTextures();
	KNNBinningV1UnbindTexture();

#if defined TIMING && defined _DEBUG
	//
	//	TIMING:
	//
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	char szString[128] = {0};
	sprintf_s( szString, "KNNBinningV1CUDA,%f\n", elapsedTime );
	//OutputDebugStringToFile( szString );
	OutputDebugString( szString );

	// Destroy the events.
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
#endif
}

void KNNBinningV1CUDA::close( void )
{
	// The KNNData has most likely changed.
	m_pKNNData->setSyncHost();
}

#pragma endregion
