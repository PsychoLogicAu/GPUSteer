#ifndef OPENSTEER_DEV_VECTOR_CUH
#define OPENSTEER_DEV_VECTOR_CUH

#include <vector>

#include <cutil_inline.h>

namespace OpenSteer
{
// Internal device vector class. Thrust's one is too bulky, requires everything to be compiled by nvcc, and throws millions of compile warnings.
template < typename T >
class dev_vector
{
	typedef typename T * iterator;
	typedef typename T const* const_iterator;
private:
	T *		m_pdMem;	// Data on the CUDA device.
	size_t	m_nSize;	// Number of elements.

	// Internal helpers.
	T * allocate( size_t const& nSize )
	{
		// Allocate device memory to hold nSize elements of type T.
		T * pdMem;
		CUDA_SAFE_CALL( cudaMalloc( (void**)&pdMem, nSize * sizeof(T) ) );
		return pdMem;
	}
	cudaError_t free( T * pdMem )
	{
		cudaError_t error;
		CUDA_SAFE_CALL( error = cudaFree( pdMem ) );
		return error;
	}

public:
	dev_vector<T>( void )					// Default - set nSize to something appropriate.
		:	m_nSize( 0 ),
			m_pdMem( NULL )
	{	}
	dev_vector<T>( size_t const& nSize )	// nSize = number of elements to reserve.
		:	m_nSize( nSize )
	{
		m_pdMem = allocate( m_nSize );
	}
	~dev_vector<T>( void )					// Destructor.
	{
		free( m_pdMem );
	}

	dev_vector<T>( dev_vector<T> & other )	// Copy constructor.
	{
		if( m_nSize != other.m_nSize )
		{
			free( m_pdMem );
			m_nSize = other.m_nSize;
			m_pdMem = allocate( m_nSize );
		}
		
		CUDA_SAFE_CALL( cudaMemcpy( m_pdMem, other.m_pdMem, m_nSize * sizeof(T), cudaMemcpyDeviceToDevice ) );
	}

	dev_vector<T>( std::vector<T> & stlVec )	// Copy constructor.
	{
		if( m_nSize != stlVec.size() )
		{
			free( m_pdMem );
			m_nSize = stlVec.size();
			m_pdMem = allocate( m_nSize );
		}

		CUDA_SAFE_CALL( cudaMemcpy( m_pdMem, &stlVec[0], m_nSize * sizeof(T), cudaMemcpyHostToDevice ) );
	}

	dev_vector<T> & operator=( const dev_vector<T> & other )	// Assignment from other dev_vector<T>
	{
		if( m_nSize != other.m_nSize )
		{
			free( m_pdMem );
			m_nSize = other.m_nSize;
			m_pdMem = allocate( m_nSize );
		}
		
		CUDA_SAFE_CALL( cudaMemcpy( m_pdMem, other.m_pdMem, m_nSize * sizeof(T), cudaMemcpyDeviceToDevice ) );
		return *this;
	}

	dev_vector<T> & operator=( const std::vector<T> & stlVec )	// Assignment from stl vector<T>.
	{
		if( m_nSize != stlVec.size() )
		{
			free( m_pdMem );
			m_nSize = stlVec.size();
			m_pdMem = allocate( m_nSize );
		}

		CUDA_SAFE_CALL( cudaMemcpy( m_pdMem, &stlVec[0], m_nSize * sizeof(T), cudaMemcpyHostToDevice ) );
		return *this;
	}

	void resize( size_t const& nSize )			// Resize the device memory and copy over current contents.
	{
		if( m_nSize == nSize )
			return;

		// Allocate device memory.
		T * pdNewMem = allocate( nSize );
		// Copy old to new.
		CUDA_SAFE_CALL( cudaMemcpy( pdNewMem, m_pdMem, m_nSize * sizeof(T), cudaMemcpyDeviceToDevice ) );
		// Set new size.
		m_nSize = nSize;
		// Free old.
		free( m_pdMem );
		m_pdMem = pdNewMem;
	}

	void clear( void )
	{
		free( m_pdMem );
		m_nSize = 0;
	}

	size_t size( void ) const
	{
		return m_nSize;
	}

	iterator begin( void )							// Return a pointer to the first element in the vector.
	{
		return m_pdMem;
	}

	iterator end( void )								// Return a pointer to one past the last element in the vector.
	{
		return m_pdMem + m_nSize;
	}

	const_iterator begin( void ) const
	{
		return m_pdMem;
	}

	const_iterator end( void ) const
	{
		return m_pdMem + m_nSize;
	}


};	// class dev_vector
};	// namespace OpenSteer

#endif
