#ifndef OPENSTEER_GLOBALS_H
#define OPENSTEER_GLOBALS_H

#define SAFE_DELETE( x )		{ if( x ){ delete x; x = NULL; } }
#define SAFE_DELETE_ARRAY( x )	{ if( x ){ delete [] x; x = NULL; } }

#endif
