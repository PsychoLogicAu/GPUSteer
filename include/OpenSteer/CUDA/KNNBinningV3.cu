#include "KNNBinningV3.cuh"

using namespace OpenSteer;

#include "KNNBinData.cuh"

#include <thrust/sort.h>

//#define TIMING

// Kernel file function prototypes.
extern "C"
{
	// Bind texCellIndices to the cudaArray.
	__host__ void KNNBinningV3BindTexture( cudaArray * pCudaArray );
	__host__ void KNNBinningV3UnbindTexture( void );

	__host__ void KNNBinningV3KernelBindTextures(		uint const*		pdBCellStart,
														uint const*		pdBCellEnd,
														uint const*		pdBIndices,
														float4 const*	pdBPosition,
														float4 const*	pdBPositionSorted,
														uint const		numCells,
														uint const		numB,
														uint const		neighborsPerCell,
														uint const*		pdCellNeighbours
														);

	__host__ void KNNBinningV3KernelUnbindTextures( void );
	__host__ void KNNBinningV3ReorderDBBindTextures(	float4 const*	pdPosition,
													uint const		numAgents );
	__host__ void KNNBinningV3ReorderDBUnbindTextures( void );

	__global__ void KNNBinningV3BuildDB(					float4 const*	pdPosition,				// In:	Positions of each agent.
														size_t *		pdAgentIndices,			// Out:	Indices of each agent.
														size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
														size_t const	numAgents
														);

	__global__ void KNNBinningV3ReorderDB(				uint const*		pdAgentIndices,		// In: (sorted) agent index.
														uint const*		pdCellIndices,		// In: (sorted) cell index agent is in.

														float4 *		pdPositionSorted,	// Out: Sorted agent positions.

														uint *			pdCellStart,		// Out: Start index of this cell in pdCellIndices.
														uint *			pdCellEnd,			// Out: End index of this cell in pdCellIndices.

														size_t const	numAgents
														);

	__global__ void KNNBinningV3Kernel(					// Group A
														float4 const*	pdAPositionSorted,			// In:	Sorted group A positions.

														uint const*		pdAIndices,					// In:	Sorted group A indices
														uint const*		pdACellIndices,				// In:	Sorted group A cell indices.

														// Cell neighbor info.
														uint const		neighborsPerCell,			// In:	Number of neighbors per cell in the pdCellNeighbors array.
														uint const		radius,						// In:	Search radius (in cells) to consider.
														uint const		numCells,
														uint const*		pdCellNeighbours,


														// Output data.
														uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
														float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.

														uint const		k,							// In:	Number of neighbors to consider.
														uint const		numA,						// In:	Size of group A.
														uint const		numB,						// In:	Size of group B.
														bool const		groupWithSelf,				// In:	Are we testing this group with itself? (group A == group B)
														bool const		bSeed
														);
}

#pragma region KNNBinningV3UpdateDBCUDA

KNNBinningV3UpdateDBCUDA::KNNBinningV3UpdateDBCUDA( BaseGroup * pGroup, KNNBinDataV3 * pKNNBinData )
:	AbstractCUDAKernel( NULL, 1.f, 0 ),
	m_pGroup( pGroup ),
	m_pKNNBinData( pKNNBinData )
{
	// Nothing to do.
}

void KNNBinningV3UpdateDBCUDA::init( void )
{
	// Bind the lookup texture.
	KNNBinningV3BindTexture( m_pKNNBinData->pdCellIndexArray() );
}

void KNNBinningV3UpdateDBCUDA::run( void )
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
	KNNBinningV3BuildDB<<< grid, block >>>( pdPosition, pdAgentIndicesSorted, pdCellIndices, numAgents );
	cutilCheckMsg( "KNNBinningV3BuildDB failed." );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	// Copy pdCellIndices to  pdCellIndicesSorted.
	CUDA_SAFE_CALL( cudaMemcpy( pdCellIndicesSorted, pdCellIndices, numAgents * sizeof(uint), cudaMemcpyDeviceToDevice ) );

	// Sort pdAgentIndicesSorted on pdCellIndicesSorted using thrust.
	thrust::sort_by_key(	thrust::device_ptr<uint>( pdCellIndicesSorted ),
							thrust::device_ptr<uint>( pdCellIndicesSorted + numAgents ),
							thrust::device_ptr<uint>( pdAgentIndicesSorted ) );

	// Set all cells to empty.
	CUDA_SAFE_CALL( cudaMemset( pdCellStart, 0xffffffff, m_pKNNBinData->getNumCells() * sizeof(uint) ) );

	// Bind the textures.
	KNNBinningV3ReorderDBBindTextures( pdPosition, numAgents );

	// Call KNNBinningReorderDB to re-order the data in the DB.
	KNNBinningV3ReorderDB<<< grid, block >>>( pdAgentIndicesSorted, pdCellIndicesSorted, pdPositionSorted, pdCellStart, pdCellEnd, numAgents );	cutilCheckMsg( "KNNBinningReorderDB failed." );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	KNNBinningV3ReorderDBUnbindTextures();

#if defined TIMING
	//
	//	TIMING:
	//
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	char szString[128] = {0};
	sprintf_s( szString, "KNNBinningV3UpdateDBCUDA,%f\n", elapsedTime );
	//OutputDebugStringToFile( szString );
	OutputDebugString( szString );

	// Destroy the events.
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
#endif
}

void KNNBinningV3UpdateDBCUDA::close( void )
{
	// Unbind the texture.
	KNNBinningV3UnbindTexture();

	// The AgentGroup's database has now changed.
	m_pGroup->SetSyncHost();
}
#pragma endregion


#pragma region KNNBinningV3

KNNBinningV3CUDA::KNNBinningV3CUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinDataV3 * pKNNBinData, BaseGroup * pOtherGroup, uint const searchRadius )
:	AbstractCUDAKernel( pAgentGroup, 1.f, 0 ),
	m_pKNNData( pKNNData ),
	m_pKNNBinData( pKNNBinData ),
	m_pOtherGroup( pOtherGroup ),
	m_searchRadius( searchRadius )
{
}

void KNNBinningV3CUDA::init( void )
{
	// Bind the cell indices texture.
	//KNNBinningV3CUDABindTexture( m_pKNNBinData->pdCellIndexArray() );
}

void KNNBinningV3CUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather the required data.
	float4 const*		pdAPositionSorted		= m_pAgentGroup->GetKNNDatabase().pdPositionSorted();
	uint const*			pdAIndices				= m_pAgentGroup->GetKNNDatabase().pdAgentIndicesSorted();
	uint const*			pdACellIndices			= m_pAgentGroup->GetKNNDatabase().pdCellIndicesSorted();

	float4 const*		pdBPosition				= m_pOtherGroup->pdPosition();
	float4 const*		pdBPositionSorted		= m_pOtherGroup->GetKNNDatabase().pdPositionSorted();
	uint const*			pdBIndices				= m_pOtherGroup->GetKNNDatabase().pdAgentIndicesSorted();
	uint const*			pdBCellIndices			= m_pOtherGroup->GetKNNDatabase().pdCellIndicesSorted();

	uint const*			pdBCellStart			= m_pOtherGroup->GetKNNDatabase().pdCellStart();
	uint const*			pdBCellEnd				= m_pOtherGroup->GetKNNDatabase().pdCellEnd();

	uint const*			pdCellNeighbors			= m_pKNNBinData->pdCellNeighbors();
	uint const&			neighborsPerCell		= m_pKNNBinData->neighborsPerCell();
	uint const&			numCells				= m_pKNNBinData->getNumCells();

	uint *				pdKNNIndices			= m_pKNNData->pdKNNIndices();
	float *				pdKNNDistances			= m_pKNNData->pdKNNDistances();

	uint const&			k						= m_pOtherGroup->GetKNNDatabase().k();
	uint const&			numA					= getNumAgents();
	uint const&			numB					= m_pOtherGroup->Size();

	bool const			groupWithSelf			= m_pAgentGroup == m_pOtherGroup;
	bool const&			bSeed					= m_pKNNData->seedable();

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
	KNNBinningV3KernelBindTextures(	pdBCellStart, pdBCellEnd, pdBIndices, pdBPosition, pdBPositionSorted, numCells, numB, neighborsPerCell, pdCellNeighbors );

	// Call the KNNBinning kernel.
	KNNBinningV3Kernel<<< grid, block, shMemSize >>>(	pdAPositionSorted,
														pdAIndices, pdACellIndices, 
														neighborsPerCell, m_searchRadius,
														numCells, pdCellNeighbors,
														pdKNNIndices, pdKNNDistances, k,
														numA, numB, groupWithSelf,
														bSeed
														);
	cutilCheckMsg( "KNNBinningV3Kernel failed." );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	KNNBinningV3KernelUnbindTextures();

#if defined TIMING
	//
	//	TIMING:
	//
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	char szString[128] = {0};
	sprintf_s( szString, "KNNBinningV3CUDA,%f\n", elapsedTime );
	//OutputDebugStringToFile( szString );
	OutputDebugString( szString );

	// Destroy the events.
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
#endif
}

void KNNBinningV3CUDA::close( void )
{
	// Data will now be seedable.
	m_pKNNData->seedable( true );

	// The KNNData has most likely changed.
	m_pKNNData->setSyncHost();
}

#pragma endregion
