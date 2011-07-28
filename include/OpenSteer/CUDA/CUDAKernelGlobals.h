#ifndef KERNELGLOBALSCUDA_H
#define KERNELGLOBALSCUDA_H

//
// Global memory
//
#define VDATA(i)		vehicleData[i]
#define VCONST(i)		vehicleConst[i]

// VehicleData
#define STEERING(i)		VDATA(i).steering
#define SPEED(i)		VDATA(i).speed
#define VELOCITY(i)		VDATA(i).velocity()
#define POSITION(i)		VDATA(i).position

#define FORWARD(i)		VDATA(i).forward
#define UP(i)			VDATA(i).up
#define SIDE(i)			VDATA(i).side

// VehicleConst
#define MAXFORCE(i)		VCONST(i).maxForce
#define MASS(i)			VCONST(i).mass
#define MAXSPEED(i)		VCONST(i).maxSpeed
#define RADIUS(i)		VCONST(i).radius

//
// Shared memory
//
#define VDATA_SH(i)		vehicleDataShared[i]
#define VCONST_SH(i)	vehicleConstShared[i]

// VehicleData
#define STEERING_SH(i)		VDATA_SH(i).steering
#define SPEED_SH(i)			VDATA_SH(i).speed
#define VELOCITY_SH(i)		VDATA_SH(i).velocity()
#define POSITION_SH(i)		VDATA_SH(i).position

#define FORWARD_SH(i)		VDATA_SH(i).forward
#define UP_SH(i)			VDATA_SH(i).up
#define SIDE_SH(i)			VDATA_SH(i).side

// VehicleConst
#define MAXFORCE_SH(i)		VCONST_SH(i).maxForce
#define MASS_SH(i)			VCONST_SH(i).mass
#define MAXSPEED_SH(i)		VCONST_SH(i).maxSpeed
#define RADIUS_SH(i)		VCONST_SH(i).radius

#endif