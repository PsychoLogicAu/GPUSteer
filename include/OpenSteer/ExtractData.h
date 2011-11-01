#ifndef OPENSTEER_EXTRACTDATA_H
#define OPENSTEER_EXTRACTDATA_H

#include "AgentGroup.h"

namespace OpenSteer
{
	void WriteCellDensity( char const* szImageFilename, AgentGroup * pAgentGroup );
	void WriteAvgCellVelocity( char const* szImageFilename, AgentGroup * pAgentGroup );

}	// namespace OpenSteer


#endif