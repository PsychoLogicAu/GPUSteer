#ifndef KERNELGLOBALSCUDA_H
#define KERNELGLOBALSCUDA_H

#include "../VectorUtils.cu"
#include "../VehicleGroupData.cu"
//
// Global memory
//
//#define VDATA(i)		vehicleData[i]
//#define VCONST(i)		vehicleConst[i]

// VehicleData
//#define STEERING(i)		VDATA(i).steering
//#define SPEED(i)		VDATA(i).speed
//#define VELOCITY(i)		VDATA(i).velocity()
//#define POSITION(i)		VDATA(i).position
//
//#define FORWARD(i)		VDATA(i).forward
//#define UP(i)			VDATA(i).up
//#define SIDE(i)			VDATA(i).side
//
//// VehicleConst
//#define MAXFORCE(i)		VCONST(i).maxForce
//#define MASS(i)			VCONST(i).mass
//#define MAXSPEED(i)		VCONST(i).maxSpeed
//#define RADIUS(i)		VCONST(i).radius

#define STEERING(i)		pdSteering[i]
#define SPEED(i)		pdSpeed[i]
#define VELOCITY(i)		velocity( i, pdForward[i], pdSpeed[i] )
#define POSITION(i)		pdPosition[i]

#define FORWARD(i)		pdForward[i]
#define UP(i)			pdUp[i]
#define SIDE(i)			pdSide[i]

// VehicleConst
#define MAXFORCE(i)		pdMaxForce[i]
#define MASS(i)			pdMass[i]
#define MAXSPEED(i)		pdMaxSpeed[i]
#define RADIUS(i)		pdRadius[i]


//
// Shared memory
//
//#define VDATA_SH(i)		vehicleDataShared[i]
//#define VCONST_SH(i)	vehicleConstShared[i]

// VehicleData
#define STEERING_SH(i)		shSteering[i]
#define SPEED_SH(i)			shSpeed[i]
#define VELOCITY_SH(i)		VDATA_SH(i).velocity()
#define VELOCITY_SH(i)		velocity( i, shForward, shSpeed )
#define POSITION_SH(i)		shPosition[i]

#define FORWARD_SH(i)		shForward[i]
#define UP_SH(i)			shUp[i]
#define SIDE_SH(i)			shSide[i]

// VehicleConst
#define MAXFORCE_SH(i)		shMaxForce[i]
#define MASS_SH(i)			shMass[i]
#define MAXSPEED_SH(i)		shMaxSpeed[i]
#define RADIUS_SH(i)		shRadius[i]

#endif