#ifndef OPENSTEER_EXTRACTDATA_H
#define OPENSTEER_EXTRACTDATA_H

#include "AgentGroup.h"

namespace OpenSteer
{
	void WriteCellDensity( char const* szFilenamePrefix, AgentGroup * pAgentGroup, std::vector< uint > vecSelectedCells = std::vector< uint >() );
	void WriteAvgCellVelocity( char const* szFilenamePrefix, AgentGroup * pAgentGroup );

}	// namespace OpenSteer


#endif