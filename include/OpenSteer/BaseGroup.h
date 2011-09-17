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

public:
	BaseGroup( uint const k, uint const size, uint const cells )
		:	m_neighborDB( k, size, cells )
	{}
	virtual ~BaseGroup( void ) {}

	KNNDatabase &			GetKNNDatabase( void )			{ return m_neighborDB; }

	// Pure virtual methods.
	virtual uint			Size( void ) const = 0;
	virtual float3 *		pdPosition( void ) = 0;
};	// class BaseGroup
}	// namespace OpenSteer

#endif
