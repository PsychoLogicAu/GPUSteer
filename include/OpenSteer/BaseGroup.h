#ifndef OPENSTEER_BASEGROUP_H
#define OPENSTEER_BASEGROUP_H

#include "CUDA/KNNDatabase.cuh"

namespace OpenSteer
{
class BaseGroup
{
protected:
	// Nearest neighbor data.
	KNNDatabase					m_neighborDB;
	
	size_t						m_nCount;

public:
	BaseGroup( uint const k, uint const size, uint const cells )
	:	m_neighborDB( k, size, cells ),
		m_nCount( 0 )
	{}
	virtual ~BaseGroup( void ) {}

	KNNDatabase &		GetKNNDatabase( void )			{ return m_neighborDB; }
	virtual uint const&	Size( void ) const				{ return m_nCount; }

	// Pure virtual methods.
	virtual float3 *		pdPosition( void ) = 0;
	virtual float3 *		pdDirection( void ) = 0;
	virtual float *			pdSpeed( void ) = 0;
	virtual float *			pdRadius( void ) = 0;

	virtual void			SetSyncHost( void ) = 0;
};	// class BaseGroup
}	// namespace OpenSteer

#endif
